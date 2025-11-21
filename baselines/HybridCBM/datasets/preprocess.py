import os

import torch
import pandas as pd
import gzip
import json
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from clip.simple_tokenizer import SimpleTokenizer
from clip import clip
from PIL import Image


def load_classes():
    classes = []
    for data in ['CIFAR10', 'CIFAR100', 'CUB', 'DTD', 'flower',
                 'food', 'HAM10000', 'ImageNet', 'RESISC45', 'UCF101']:
        cls = list(pd.read_csv(f'./{data}/class.csv')['class'].unique())
        if data in ['aircraft']:
            cls = [f'aircraft {c}' for c in cls]
            cls.append('aircraft')
        classes.extend(cls)
    return classes


def load_assertions(file_path):
    save_path = file_path.replace('.csv.gz', '-en.csv')
    if os.path.exists(save_path):
        return pd.read_csv(save_path)
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        df = pd.read_csv(f, sep='\t', names=['uri', 'relation', 'start', 'end', 'info'],
                         usecols=['relation', 'start', 'end', 'info'], dtype=str)
    df = df.loc[df['start'].str.startswith('/c/en/') & df['end'].str.startswith('/c/en/')]
    df['start'] = df['start'].map(lambda x: x.replace('/c/en/', ''))
    df['end'] = df['end'].map(lambda x: x.replace('/c/en/', ''))
    df['relation'] = df['relation'].map(lambda x: x.split('/')[-1])
    df.to_csv(save_path, index=False)
    return df


def collect_conceptnet_concepts():
    translator_path = './ConceptTranslator'
    raw_data = load_assertions("./ConceptTranslator/conceptnet-assertions-5.7.0.csv.gz")
    raw_data = raw_data.loc[raw_data['info'].map(lambda x: 'surfaceText' in x)]
    raw_data['info'] = raw_data['info'].map(lambda x: json.loads(x)['surfaceText'].replace('[', '').replace(']', ''))
    concepts = raw_data.loc[raw_data['info'].map(lambda x: len(x.split()) <= 15)]['info'].unique()
    os.makedirs(translator_path, exist_ok=True)
    pd.DataFrame({'concept': concepts}).to_json(os.path.join(translator_path, 'conceptNet.json'), index=False)


def collect_generated_concepts():
    cub = json.load(open('./CUB/concepts/class2concepts.json', 'r'))
    cifar10 = json.load(open('./CIFAR10/concepts/class2concepts.json', 'r'))
    cifar100 = json.load(open('./CIFAR100/concepts/class2concepts.json', 'r'))
    aircraft = json.load(open('./aircraft/concepts/class2concepts.json', 'r'))
    DTD = json.load(open('./DTD/concepts/class2concepts.json', 'r'))
    flower = json.load(open('./flower/concepts/class2concepts.json', 'r'))
    food = json.load(open('./food/concepts/class2concepts.json', 'r'))
    HAM10000 = json.load(open('./HAM10000/concepts/class2concepts.json', 'r'))
    ImageNet = json.load(open('./ImageNet/concepts/class2concepts.json', 'r'))
    RESISC45 = json.load(open('./RESISC45/concepts/class2concepts.json', 'r'))
    UCF101 = json.load(open('./UCF101/concepts/class2concepts.json', 'r'))

    concepts = []
    for d in [cub, cifar10, cifar100, aircraft, DTD, flower, food, HAM10000, ImageNet, RESISC45, UCF101]:
        for dd in d.values():
            concepts.extend(dd)
    concepts = list(set(concepts))
    concepts = pd.DataFrame({'concept': concepts})
    translator_path = './ConceptTranslator'
    os.makedirs(translator_path, exist_ok=True)
    concepts.to_json(os.path.join(translator_path, 'generatedConcepts.json'), index=False)


def tokenize(context_length: int = 77):
    if os.path.exists('./ConceptTranslator/conceptsBank_tokens.pt'):
        return torch.load('./ConceptTranslator/conceptsBank_tokens.pt')
    coco = pd.read_json('./ConceptTranslator/subCOCO.json')
    conceptNet = pd.read_json('./ConceptTranslator/conceptNet.json')
    generated = pd.read_json('./ConceptTranslator/generatedConcepts.json')
    concepts = pd.concat([coco, conceptNet, generated]).reset_index(drop=True)
    concepts.to_json('./ConceptTranslator/conceptsBank.json', index=False)
    print("Total concepts: ", len(concepts))

    tokenizer = SimpleTokenizer()
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]

    all_tokens = []
    for text in tqdm(concepts['concept']):
        try:
            all_tokens.append([sot_token] + tokenizer.encode(text) + [eot_token])
        except Exception as e:
            print(f'Error: {e} for text: {text}')

    result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)
    all_len = []
    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            tokens = tokens[:context_length]
            tokens[-1] = eot_token
        all_len.append(tokens.__len__())
        result[i, :len(tokens)] = torch.tensor(tokens)
    all_len = torch.tensor(all_len).float()
    print("Max length: ", all_len.max())
    print("Min length: ", all_len.min())
    print("Avg length: ", all_len.mean())
    print("Std length: ", all_len.std())
    print("Lambda Length", int(all_len.mean() + all_len.std() * 10))
    if not os.path.exists('./ConceptTranslator/conceptsBank_tokens.pt'):
        torch.save(result, './ConceptTranslator/conceptsBank_tokens.pt')


def json_to_csv():
    for p in os.listdir('selected_concepts'):
        p = os.path.join('selected_concepts', p)
        if p.endswith('.json'):
            df = pd.read_json(p).melt(value_name='concept', var_name='class')
            df.to_csv(p.replace('.json', '.csv'), index=False)


@torch.no_grad()
def concept(clip_model_type='ViT-L/14', device=4):
    # collect_conceptnet_concepts()
    # collect_generated_concepts()
    # json_to_csv()
    tokens = tokenize()
    device = torch.device(device)
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)
    clip_model_name = clip_model_type.replace('/', '_')
    out_path = f"ConceptTranslator/ConceptBank_{clip_model_name}.pkl"
    if os.path.exists(out_path):
        print(f"File exists: {out_path}")
        return

    class _Dataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __getitem__(self, index):
            return self.data[index]

        def __len__(self):
            return len(self.data)

    loader = DataLoader(_Dataset(tokens), batch_size=1024, shuffle=False, num_workers=16)
    all_embeddings = []
    all_tokens = []
    for tokens in tqdm(loader):
        tokens = tokens.to(device)
        embeddings = clip_model.encode_text(tokens).cpu()
        all_embeddings.append(embeddings)
        all_tokens.append(tokens)
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_tokens = torch.cat(all_tokens, dim=0)
    torch.save({"embedding": all_embeddings, "tokens": all_tokens}, out_path)


@torch.no_grad()
def coco(clip_model_type='ViT-L/14', device=4):  # 566747
    device = torch.device(device)
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)
    clip_model_name = clip_model_type.replace('/', '_')
    out_path = f"COCO/COCO_{clip_model_name}_train.pkl"
    json_path = 'COCO/coco_data.json'
    if os.path.exists(out_path):
        print(f"File exists: {out_path}")
        return
    if os.path.exists(json_path):
        data = pd.read_json(json_path)
        print(f"File loaded: {json_path}")
    else:
        with open('COCO/train_caption.json', 'r') as f:
            captions = json.load(f)
        print(f"{len(captions)} captions loaded from json ")
        data = []
        for d in tqdm(captions):
            img_id = d["image_id"]
            caption = d['caption']
            filename = f"COCO/train2014/COCO_train2014_{int(img_id):012d}.jpg"
            if not os.path.isfile(filename):
                filename = f"COCO/val2014/COCO_val2014_{int(img_id):012d}.jpg"
            if not os.path.isfile(filename):
                print(f"File not found: {filename}")
                continue
            data.append({"image_id": img_id, "image": filename, "caption": caption})
        data = pd.DataFrame(data)
        data.to_json('COCO/coco_data.json', index=False)

    class _Dataset(Dataset):
        def __init__(self, data, transform=None):
            self.data = data
            self.transform = transform
            self.tokenize = clip.tokenize

        def __getitem__(self, index):
            data = self.data[index]
            img = Image.open(data["image"]).convert('RGB')
            img = self.transform(img)
            tokens = self.tokenize(data["caption"], truncate=True)[0]
            return img, tokens

        def __len__(self):
            return len(self.data)

    loader = DataLoader(_Dataset(data.to_dict(orient='records'), preprocess), batch_size=1024, shuffle=False,
                        num_workers=16)
    all_embeddings = []
    all_tokens = []
    for image, tokens in tqdm(loader):
        image = image.to(device)
        embedding = clip_model.encode_image(image).cpu()
        all_embeddings.append(embedding)
        all_tokens.append(tokens)
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_tokens = torch.cat(all_tokens, dim=0)
    torch.save({"embedding": all_embeddings, "tokens": all_tokens}, out_path)


if __name__ == "__main__":
    # collect_conceptnet_concepts()
    # collect_generated_concepts()
    # json_to_csv()
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=4)
    parser.add_argument('--data', type=str, default='coco')
    args = parser.parse_args()
    if args.data == 'coco':
        coco(clip_model_type='RN50', device=args.device)
    else:
        concept(clip_model_type='RN50', device=args.device)
