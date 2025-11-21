import torch
import torch.nn as nn
from .translator import ConceptTranslator
from .clip import ClipEncoder


class ClipCap(nn.Module):
    def __init__(self, model_name: str = 'ViT-B/32', encoder=None, use_txt_norm=False,
                 use_img_norm=False, weights_path='./coco_prefix.pt'):
        super(ClipCap, self).__init__()
        if encoder is None:
            self.encoder = ClipEncoder(model_name=model_name, use_txt_norm=use_txt_norm, use_img_norm=use_img_norm, )
        else:
            self.encoder = encoder
        self.preprocess = self.encoder.preprocess
        self.translator = ConceptTranslator(clip_model=model_name)

        checkpoint = torch.load(weights_path, map_location=torch.device('cpu'))
        self.translator.load_state_dict(checkpoint)
        self.translator.eval()

    def text_encode(self, text, batch_size=128) -> torch.Tensor:
        return self.encoder.encode_text(text, batch_size=batch_size)

    def image_encode(self, image: torch.Tensor) -> torch.Tensor:
        return self.encoder.encode_image(image)

    def text_decode(self, clip_features, entry_length=30, temperature=1, batch_size=128):
        return self.translator.decode(clip_features, entry_length, temperature, batch_size)

    def image_caption_from_concepts(self, path_pic, concepts=('',)):
        image_features = self.image_encode(path_pic)
        text_features = self.text_encode(concepts)
        sim = image_features @ text_features.T.float()
        sim = (sim * 100).softmax(dim=-1)
        prefix_embedding = sim @ text_features.float()
        prefix_embedding /= prefix_embedding.norm(dim=-1, keepdim=True)
        generated_text = self.text_decode(prefix_embedding)
        print(generated_text)


def freeze(model):
    for param in model.parameters():
        param.requires_grad = False


def unfreeze(model):
    for param in model.parameters():
        param.requires_grad = True
