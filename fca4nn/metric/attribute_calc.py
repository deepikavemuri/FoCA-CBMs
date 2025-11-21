import torch
from tqdm.auto import tqdm


class AttributeMetric:
    def __init__(self, model, test_dataloader, is_foca=True, args=None):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.test_dataloader = test_dataloader
        self.is_foca = is_foca
        self.args = args

        self.attr_preds = []
        self.attrs_labels = []

    def get_attr_preds_cbm(self):
        with torch.no_grad():
            for batch in self.test_dataloader:
                imgs, _, attrs = batch
                concepts, _ = self.model(imgs.to(self.device))

                self.attr_preds.append(concepts.detach().cpu().numpy().tolist())
                self.attrs_labels.append(attrs.detach().cpu().numpy().tolist())

    def get_attr_preds_foca(self):
        with torch.no_grad():
            for batch in tqdm(
                self.test_dataloader,
            ):
                _, imgs, _, attrs_present, _ = batch
                attrs_preds, _ = self.model(imgs.to(self.device))
