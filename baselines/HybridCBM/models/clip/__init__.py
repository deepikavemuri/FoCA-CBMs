import torch
import torch.nn as nn
from .clip import tokenize, load
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class _Dataset(Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform

    def __getitem__(self, index):
        img = self.images[index]
        if self.transform:
            img = Image.open(self.images[index]).convert('RGB')
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.images)


class ClipEncoder(nn.Module):
    def __init__(self, model_name: str = 'ViT-B/32',
                 clip_ckpt=None,
                 use_txt_norm=False,
                 use_img_norm=False, ):
        super(ClipEncoder, self).__init__()
        self.model_name = model_name
        self.clip, self.preprocess = load(model_name, device='cpu', jit=False)
        self.embedding_dim = self.clip.visual.output_dim
        self.input_resolution = self.clip.visual.input_resolution
        if clip_ckpt:
            self.clip.load_state_dict(torch.load(clip_ckpt))
        self.clip = self.clip.eval()
        self.tokenizer = tokenize
        self.use_txt_norm = use_txt_norm
        self.use_img_norm = use_img_norm

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
            return torch.device('cpu')

    @torch.no_grad()
    def encode_text(self, text, batch_size=1, normalize=False) -> torch.Tensor:
        if isinstance(text, str):
            text = [text]
        loader = DataLoader(_Dataset(self.tokenizer(text)), batch_size=batch_size, shuffle=False)
        text_features = []
        for batch in tqdm(loader, desc='Encoding text'):
            batch = batch.to(self.device)
            text_feature = self.clip.encode_text(batch).float().cpu()
            text_features.append(text_feature)
        text_features = torch.cat(text_features, dim=0)
        if self.use_txt_norm or normalize:
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    @torch.no_grad()
    def encode_image(self, images, batch_size=1, normalize=False) -> torch.Tensor:
        if not isinstance(images, torch.Tensor):
            if isinstance(images, str):
                images = [images]
            dataset = _Dataset(images, self.preprocess)
        else:
            dataset = _Dataset(images)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        image_features = []
        for batch in tqdm(loader, desc='Encoding image'):
            batch = batch.to(self.device)
            image_features.append(self.clip.encode_image(batch).float().cpu())
        image_features = torch.cat(image_features, dim=0)
        if self.use_img_norm or normalize:
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features
