import glob

import pandas as pd
import torch
import os
import argparse
import sys
import clip
import json
import random
import torch.distributed as dist

from typing import Tuple

from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from models.translator import ConceptTranslator

from nltk.corpus import wordnet


class ConceptDataset(Dataset):
    def __init__(self, data_path: str, aug: bool = False, use_embedding=False):
        self.clip_tokenizer = clip.tokenize
        self.max_seq_len = 20
        self.aug = aug
        if use_embedding:
            data = torch.load(data_path, map_location='cpu', weights_only=True)
            self.embeddings = data['embedding']
            self.embeddings = self.embeddings / self.embeddings.norm(dim=-1, keepdim=True)
            self.captions = data['tokens'].squeeze().type(torch.int64)
            print(f'Embeddings loaded from {data_path} {self.embeddings.shape}')
            print(f'Tokens loaded from {data_path} {self.captions.shape}')
        else:
            self.embeddings = None
            if aug:
                self.captions = pd.read_json(data_path)['concept'].tolist()
            else:
                cache_path = os.path.join(os.path.dirname(data_path), 'conceptsBank_tokens.pt')
                if os.path.exists(cache_path):
                    self.captions = torch.load(cache_path)
                else:
                    with open(data_path, 'r') as f:
                        captions = json.load(f)
                    self.captions = []
                    batch_size = 512
                    for start in range(0, len(captions), batch_size):
                        end = min(start + batch_size, len(captions))
                        self.captions.extend(captions[start:end])
                    self.captions = torch.stack(self.captions, dim=0)
                    torch.save(self.captions, cache_path)

    def __len__(self) -> int:
        return len(self.captions)

    def tokenize(self, text: str) -> Tensor:
        return self.clip_tokenizer(text, truncate=True).type(torch.int64)[0]

    def pad_tokens(self, tokens):
        tokens = tokens.clone()
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64)))
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
        return tokens

    def synonym_replace(self, text, p=0.1) -> str:
        if random.random() <= p:
            words = text.split(' ')
            # Select a random word that has at least one synonym
            eligible_words = [word for word in words if wordnet.synsets(word)]
            if not eligible_words:
                return text  # Return the original text if no eligible words found

            word = random.choice(eligible_words)
            synonyms = [lem.name().replace('_', ' ') for syn in wordnet.synsets(word) for lem in syn.lemmas() if
                        lem.name() != word]

            if synonyms:
                new_word = random.choice(synonyms)
                # Replace only the first occurrence of the word in the text
                first_occurrence_index = words.index(word)
                words[first_occurrence_index] = new_word

            text = ' '.join(words)
        return text

    def random_swap_tockens(self, tokens, p=0.1):
        if random.random() <= p:
            idx1, idx2 = random.sample(range(len(tokens)), 2)
            tokens[idx1], tokens[idx2] = tokens[idx2], tokens[idx1]
        return tokens

    def textual_noise_injection(self, text, p=0.5, error_ratio=.1):
        if random.random() <= p:
            characters = list(text)
            num_errors = int(len(characters) * error_ratio)  # Define how many errors to introduce
            error_type = random.choice(['replace', 'swap', 'delete'])  # Choose an error type

            for _ in range(num_errors):
                position = random.randint(0, len(characters) - 1)  # Position for the error

                if error_type == 'replace':
                    characters[position] = random.choice('abcdefghijklmnopqrstuvwxyz')
                elif error_type == 'swap' and len(characters) > 1:
                    swap_position = (position + 1) % len(characters)
                    characters[position], characters[swap_position] = characters[swap_position], characters[position]
                elif error_type == 'delete':
                    del characters[position]

            text = ''.join(characters)
        return text

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        text_or_token = self.captions[item]
        if self.embeddings is not None:
            embedding = self.embeddings[item]
            tokens = self.pad_tokens(text_or_token)
            return embedding, tokens
        if self.aug:
            # text
            text = self.synonym_replace(text_or_token, p=0.5)
            text = self.textual_noise_injection(text, p=0.1, error_ratio=.1)
            original_tokens = self.tokenize(text)
        else:
            # token
            original_tokens = text_or_token
        tokens = self.pad_tokens(original_tokens)
        return original_tokens, tokens


class CocoDataset(Dataset):
    def __init__(self, data_path: str):
        self.max_seq_len = 20
        self.aug = False
        cache_path = os.path.join(data_path)
        if os.path.exists(cache_path):
            coco = torch.load(cache_path, map_location='cpu', weights_only=True)
            self.embeddings = coco['embedding']
            self.embeddings = self.embeddings / self.embeddings.norm(dim=-1, keepdim=True)
            self.tokens = coco['tokens'].squeeze().type(torch.int64)
            print(f'Embeddings loaded from {cache_path} {self.embeddings.shape}')
            print(f'Tokens loaded from {cache_path} {self.tokens.shape}')
        else:
            raise NotImplementedError

    def __len__(self) -> int:
        return len(self.embeddings)

    def pad_tokens(self, tokens):
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat([tokens, torch.zeros(padding, dtype=torch.int64)])
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
        return tokens

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        embeddings = self.embeddings[item]
        clip_tokens = self.tokens[item]
        clip_tokens = self.pad_tokens(clip_tokens)
        return embeddings, clip_tokens


class MergeDataset(Dataset):
    def __init__(self, conceptPath, cocoPath):
        self.datasets = [
            ConceptDataset(conceptPath, use_embedding=True),
            CocoDataset(cocoPath)
        ]

    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)

    def __getitem__(self, item):
        for dataset in self.datasets:
            if item < len(dataset):
                return dataset[item]
            item -= len(dataset)
        raise IndexError(f"Index {item} out of range")


def run_step(model, optimizer, scheduler, feature, tokens, loss_ce):
    outputs, _ = model(feature.float(), tokens)
    logits = outputs

    logits = logits.logits

    logits = logits[:, : -1]
    tokens = tokens.flatten()
    logits = logits.reshape(-1, logits.shape[-1])

    loss_token = loss_ce(logits, tokens)
    optimizer.zero_grad()
    loss_all = loss_token
    loss_all.backward()
    optimizer.step()
    scheduler.step()
    return logits, tokens, loss_token


def train_translator(dataset, args, clip_name="ViT-B/32",
                     lr: float = 1e-5, warmup_steps: int = 1000, output_dir: str = ".", output_prefix: str = ""):
    # device = torch.device('cuda:1')
    batch_size = args.batch_size
    epochs = args.epochs
    name = f"{clip_name.replace('/', '_')}-AUG_{args.augment}"
    output_dir = os.path.join(output_dir, name)
    log_path = os.path.join(output_dir, f"{name}_log.txt")
    save_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    args.is_master = args.local_rank == 0

    # set the device
    torch.cuda.set_device(args.local_rank)
    device = torch.device('cuda:' + str(args.local_rank))
    dist.init_process_group(backend='nccl', init_method='env://')
    SEED = 42
    torch.cuda.manual_seed_all(SEED)

    clip_model, preprocess = clip.load(clip_name, device=device, jit=False)
    clip_model.eval()

    model = ConceptTranslator(prefix_size=clip_model.visual.output_dim)
    loss_ce = torch.nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
    checkpoints = glob.glob(os.path.join(save_dir, '*.pt'))
    if not checkpoints:
        print(f"No checkpoints found in {save_dir}")
        start_epoch = 0
    else:
        def get_val_acc(ckpt):
            filename = os.path.basename(ckpt)
            val_acc = filename.split('-')[-1].split('.pt')[0]
            return float(val_acc)

        checkpoint_path = max(checkpoints, key=get_val_acc)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        start_epoch = int(checkpoint_path.split('-')[-1].split('.pt')[0]) + 1
        print(f"Loading checkpoint from {checkpoint_path}, start from epoch {start_epoch}")

    model.to(device)
    model = DDP(
        model,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        find_unused_parameters=False
    )

    optimizer = AdamW(model.parameters(), lr=lr)
    sampler = DistributedSampler(dataset)
    train_dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, drop_last=True)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs * len(train_dataloader)
    )
    best_acc = 0

    for epoch in range(epochs):
        if epoch < start_epoch:
            for _ in range(len(train_dataloader)):
                optimizer.step()
                scheduler.step()
            continue
        print(f"Start from Epoch {epoch}")
        loss_token_save, ac_save, epoch_acc = 0, 0, []
        sys.stdout.flush()
        if args.is_master:
            print(f">>> Training epoch {epoch}")
            progress = tqdm(total=int(len(train_dataloader) / 10), desc=f'{epoch}/{epochs} {output_prefix}')
        else:
            progress = None
        dist.barrier()
        for idx, data in enumerate(train_dataloader):
            tokens = data[1].to(device)
            if isinstance(dataset, MergeDataset):
                feature = data[0].to(device)
            else:
                with torch.no_grad():
                    feature = clip_model.encode_text(data[0].to(device))
                    feature /= feature.norm(dim=-1, keepdim=True)
            logits, tokens, loss_token = run_step(model, optimizer, scheduler, feature, tokens, loss_ce)
            if progress is not None:
                ac = ((logits.argmax(1) == tokens) * (tokens > 0)).sum() / (tokens > 0).sum()
                epoch_acc.append(ac.item())
                if (idx + 1) % 10 == 0:
                    progress.set_postfix({"loss_token": loss_token_save / 10.0, "acc_token": ac_save / 10.0})
                    progress.update()
                    loss_token_save, ac_save = 0, 0
                else:
                    loss_token_save += loss_token.item()
                    ac_save += ac.item()

        if args.is_master:
            epoch_acc = sum(epoch_acc) / len(epoch_acc)

            with open(log_path, 'a+') as f:
                f.writelines('epoch ' + str(epoch) + ': ' + progress.postfix + '\r\n')
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(
                        model.module.state_dict(),
                        os.path.join(output_dir, f"{output_prefix}-best.pt"),
                    )
                    f.writelines('-------------------------- best model saved --------------------------\r\n')
            progress.close()
            if epoch % args.save_every == 0 or epoch == epochs - 1:
                if len(os.listdir(save_dir)) > 10:
                    os.remove(os.path.join(save_dir, sorted(os.listdir(save_dir))[0]))
                torch.save(
                    model.module.state_dict(),
                    os.path.join(save_dir,
                                 f"{output_prefix}-"
                                 f"{clip_name.replace('/', '_')}"
                                 f"-AUG_{args.augment}-{epoch:03d}.pt"),
                )
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', default='./weights/translators', help='output directory')
    parser.add_argument('--clip_model', default='ViT-L/14', help='clip model name')
    parser.add_argument('--output_prefix', default='translators', help='prefix for saved filenames')
    parser.add_argument('--augment', default=False, action='store_true')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--save_every', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()
    args.local_rank = int(os.environ['LOCAL_RANK'])

    dataset = MergeDataset(f"./datasets/ConceptTranslator/ConceptBank_{args.clip_model.replace('/', '_')}.pkl",
                           f"./datasets/COCO/COCO_{args.clip_model.replace('/', '_')}_train.pkl")
    train_translator(dataset, args, clip_name=args.clip_model, output_dir=args.out_dir,
                     output_prefix=args.output_prefix)
