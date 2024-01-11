import argparse
import itertools
import json
import os
from functools import partial

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

from PIL import Image 

multiple_choices = ['A', 'B', 'C', 'D', 'E']

ds_collections = {
    'scienceqa_test_img': {
        'test': 'playground/data/eval/scienceqa/scienceqa_test_img.jsonl',
    }
}


def collate_fn(batches, pad_token_id):

    input_tokens = [_['input_tokens'] for _ in batches]
    target_lengths = [_['target_lengths'] for _ in batches]
    answers = [_['answer'] for _ in batches]
    images = [_['image'] for _ in batches]

    chunk_sizes = [len(_) for _ in input_tokens] # num of choices

    input_tokens = [_ for _ in itertools.chain.from_iterable(input_tokens)] # split each chunk 

    max_lengths = max([len(_) for _ in input_tokens])
    input_tokens = [_ + [pad_token_id] * (max_lengths - len(_))
                    for _ in input_tokens] # "right" pad
    input_tokens = torch.LongTensor(input_tokens)
    images = torch.stack([img for i, img in enumerate(images) for _ in range(chunk_sizes[i])])

    attention_mask = 1 - input_tokens.eq(pad_token_id).float()

    return images, input_tokens, attention_mask, target_lengths, answers, chunk_sizes


class MultipleChoiceDataste(torch.utils.data.Dataset):

    def __init__(self, test, prompt, tokenizer):
        self.datas = open(test).readlines()
        self.prompt = prompt
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):

        data = json.loads(self.datas[idx].strip())
        image = data['image']
        image = Image.open(image)
        image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0].half()

        hint = data['hint'] if data['hint'] else 'N/A'
        question = data['question']

        choices = data['choices']
        choice_list = []
        for i, c in enumerate(choices):
            choice_list.append('{}. {}'.format(multiple_choices[i], c))
        choice_txt = '\n'.join(choice_list)

        prompt = self.prompt.format(hint, question, choice_txt)

        prompt_tokens = [
            tokenizer_image_token(prompt + f" {_}", self.tokenizer, IMAGE_TOKEN_INDEX)
            for _ in multiple_choices[:len(choices)]
        ]
        # prompt_tokens = self.tokenizer(prompt).input_ids
        # target_tokens = [
        #     self.tokenizer(' ' + _).input_ids
        #     for _ in multiple_choices[:len(choices)]
        # ]

        return {
            'image': image,
            'input_tokens': prompt_tokens,
            'target_lengths': [
            len(self.tokenizer(' ' + _).input_ids) - 2 # bos和空格
            for _ in multiple_choices[:len(choices)]
        ],
            'answer': data['answer'],
        }


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size,
                                                      self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    args = parser.parse_args()

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )

    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, device_map="cuda")

    model = model.eval()
    prompt = 'USER: <image>\n ASSISTANT: Context: {}\nQuestion: {}\nOptions: {}\nAnswer with the option\'s letter from the given choices directly. answer:'

    dataset = MultipleChoiceDataste(test=ds_collections[args.dataset]['test'],
                                    prompt=prompt,
                                    tokenizer=tokenizer)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=InferenceSampler(len(dataset)),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=partial(collate_fn, pad_token_id=IGNORE_INDEX),
    )

    results = []
    with torch.no_grad():
        for _, (images, input_tokens, attention_mask, target_lengths, answer,
                chunk_sizes) in tqdm(enumerate(dataloader), total=len(dataloader)):
            
            images = images.cuda()
            input_ids=input_tokens.cuda()
            attention_mask=attention_mask.cuda()
            labels=input_tokens.clone().cuda()
            (
                _,
                _,
                _,
                _,
                _,
                labels
            ) = model.prepare_inputs_labels_for_multimodal(
                input_ids,
                None,
                attention_mask,
                None,
                labels,
                images
            )
            outputs = model(
                images = images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # # Flatten the tokens
            # shift_logits = shift_logits.view(-1, shift_logits.shape[-1])
            # shift_labels = shift_labels.view(-1)

            losses = torch.nn.functional.cross_entropy(shift_logits.permute(0,2,1), shift_labels,reduction='none')

            losses = losses.split(chunk_sizes, dim=0)
            labels = labels.split(chunk_sizes, dim=0)

            for loss, target_length, answer, label in zip(losses, target_lengths,
                                                   answer, labels):
                # right pad
                non_zero_indices = torch.nonzero(label[:, 1:] != IGNORE_INDEX).squeeze()
                loss = loss[:, :non_zero_indices.max()+1]

                target_loss = loss.mean(-1)
                for _ in range(len(target_length)):
                    target_loss[_] = loss[_, -target_length[_]:].mean()
                # print(target_loss)
                pred = target_loss.argmin().item()
                if pred == answer:
                    # print(pred, answer)
                    results.append(1)
                else:
                    results.append(0)

    torch.distributed.barrier()

    world_size = torch.distributed.get_world_size()
    merged_results = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(merged_results, results)

    merged_results = [_ for _ in itertools.chain.from_iterable(merged_results)]

    if torch.distributed.get_rank() == 0:
        print(f"Evaluating {args.dataset} ...")
        print(f'Acc@1: {sum(merged_results) / len(merged_results)}')

    torch.distributed.barrier()
