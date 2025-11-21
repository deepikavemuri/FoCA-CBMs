from .linearCBM import LinearCBM
from ..utils import freeze


class CaptionCBM(LinearCBM):
    def __init__(self, config, conceptbank):
        super().__init__(config, conceptbank)
        freeze(self.conceptbank)

    def training_step(self, train_batch, batch_idx):
        opt_classifier = self.optimizers()
        image, label = train_batch
        if self.config.use_normalize:
            image = image / image.norm(dim=-1, keepdim=True)
        # Training classifier
        cls_loss = self.train_classifier(image, label)
        final_loss = cls_loss
        self.manual_backward(final_loss)
        opt_classifier.step()
        opt_classifier.zero_grad()
        self.log('final_loss', final_loss)
        return final_loss
