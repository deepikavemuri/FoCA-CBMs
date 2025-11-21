import os.path

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel
from typing import Tuple
from ..clip.simple_tokenizer import SimpleTokenizer
from tqdm import tqdm


class MLP(nn.Module):
    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ConceptTranslator(nn.Module):
    def __init__(self, clip_model=None, prefix_size: int = 512):
        super(ConceptTranslator, self).__init__()
        if clip_model == "ViT-L/14":
            prefix_size = 768
        elif clip_model == "ViT-B/32":
            prefix_size = 512
        else:
            prefix_size = 1024
        self.clip_model = clip_model
        self.decoder = GPT2LMHeadModel(
            GPT2Config(
                n_layer=12,
                n_head=12,
            )
        )
        self.embedding_size = self.decoder.transformer.wte.weight.shape[1]
        self.clip_project = MLP((prefix_size, self.embedding_size))
        self.tokenizer = SimpleTokenizer()

    @property
    def device(self):
        # 检查是否有参数
        if next(self.parameters(), None) is not None:
            return next(self.parameters()).device
        # 检查是否有缓冲区
        elif next(self.buffers(), None) is not None:
            return next(self.buffers()).device
        else:
            # 默认返回CPU
            return torch.device("cpu")

    def load(self, weight_path="weights/translators"):
        checkpoint = torch.load(weight_path, map_location="cpu", weights_only=True)
        self.load_state_dict(checkpoint)
        self.eval()

    def forward(
        self, clip_features, gpt_tokens=None, project=True, attention_mask=None
    ):
        if project:
            clip_features = self.clip_project(clip_features)
            clip_features = clip_features.reshape(-1, 1, self.embedding_size)
        if gpt_tokens is not None:
            embedding_text = self.decoder.transformer.wte(gpt_tokens)
            embedding_cat = torch.cat([clip_features, embedding_text], dim=1)
        else:
            embedding_cat = clip_features
        out = self.decoder(inputs_embeds=embedding_cat, attention_mask=attention_mask)
        return out, embedding_cat

    def output_to_token(self, output, temperature):
        logits = output.logits
        logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)  # B, dim
        logits = torch.nn.functional.softmax(logits, dim=-1)
        next_token = torch.argmax(logits, -1).unsqueeze(-1)
        return next_token

    @torch.no_grad()
    def decode(self, clip_features, entry_length=30, temperature=1, batch_size=128):
        tokens = []
        for start in tqdm(range(0, clip_features.shape[0], batch_size)):
            end = min(start + batch_size, clip_features.shape[0])
            clip_feature = clip_features[start:end].to(self.device)
            clip_feature = clip_feature / clip_feature.norm(dim=-1, keepdim=True)
            outputs, embedding_cat = self.forward(clip_feature, project=True)
            bc_tokens = []
            for i in range(entry_length):
                next_token = self.output_to_token(outputs, temperature)
                outputs, embedding_cat = self.forward(
                    embedding_cat, next_token, project=False
                )
                bc_tokens.append(next_token)
            bc_tokens = torch.cat(bc_tokens, dim=1).cpu()
            tokens.append(bc_tokens)
        tokens = torch.cat(tokens, dim=0).numpy()
        output_list = []
        for token_step in list(tokens):
            output = self.untokenize(token_step)
            output_list.append(output)
        return output_list if len(output_list) > 1 else output_list[0]

    def untokenize(self, tokens):
        output = []
        for token in tokens.squeeze():
            if token > 49407:
                break
            output.append(token)
            if token.item() == 49407:
                break
        output = self.tokenizer.decode(output)
        return (
            output.replace("<|startoftext|>", "").replace("<|endoftext|>", "").strip()
        )
