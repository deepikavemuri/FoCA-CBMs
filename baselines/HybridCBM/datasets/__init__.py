from .dataloader import DataBank
from models.clip import ClipEncoder


def get_datamodule_fromconfig(config, clip_encoder=None):
    if clip_encoder is None:
        clip_encoder = ClipEncoder(model_name=config.clip_model)
        clip_encoder.eval()
        clip_encoder.to(config.device)
    return DataBank(
        data_root=config.data_root,
        exp_root=config.exp_root,
        # concept
        n_shots=config.n_shots,

        # clip
        use_img_features=config.use_img_features,
        clip_encoder=clip_encoder,

        # dataloader
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        force_compute=config.force_compute,
    )
