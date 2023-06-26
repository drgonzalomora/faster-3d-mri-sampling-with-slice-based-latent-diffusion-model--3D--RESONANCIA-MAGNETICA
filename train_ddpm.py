import numpy as np
import torch
import pytorch_lightning as pl
import os
import glob
from omegaconf import OmegaConf
from pytorch_lightning.loggers import wandb as wandb_logger
from pytorch_lightning.callbacks import ModelCheckpoint

from modules.preprocessing import BRATSDataModule
from modules.loggers import DDPMImageSampler
from modules.ddpm import DDPM
from modules.diffusion import Diffusion
from modules.sampler import ScheduleSampler

os.environ['WANDB_API_KEY'] = 'bdc8857f9d6f7010cff35bcdc0ae9413e05c75e1'

def global_seed(seed, debugging=False):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    if debugging:
        torch.backends.cudnn.deterministic = True
    
if __name__ == "__main__":
    global_seed(42)
    torch.set_float32_matmul_precision('high')
    
    # loading config file
    CONFIG_PATH = './config.yaml'
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError('Config file not found')
    
    cfg = OmegaConf.load(CONFIG_PATH)
    assert cfg.target in ['first-stage-training', 'diffusion-training', 'inference']
    
    # logger
    logger = wandb_logger.WandbLogger(
        project='ddpm-brats', 
        name='Train BRATS'
    )
    
    # data module
    datamodule = BRATSDataModule(**cfg.data.first_stage)
    
    sampler = ScheduleSampler(
        T=cfg.diffusion.T,
        batch_size=cfg.data.first_stage.batch_size,
        sampler=cfg.diffusion.schedule_sampler,
        memory_span=cfg.diffusion.loss_memory_span
    )

    diffusion = Diffusion(**cfg.diffusion)

    # model
    model = DDPM(
        unet_config = cfg.unet,
        diffusion   = diffusion,
        sampler     = sampler,
        use_ema     = False,
        clamp       = True,
        lr          = 5e-05,
        weight_decay = 1e-07
    )
    
    callbacks = []
    callbacks.append(
        DDPMImageSampler(n_samples=1, every_n_epochs=10)
    )
    
    callbacks.append(
        ModelCheckpoint(
            **cfg.callbacks.checkpoint,
            filename='ddpm-{epoch}'
        )
    )
    
    # training
    trainer = pl.Trainer(
        logger=logger,
        # strategy="ddp",
        # devices=4,
        # num_nodes=2,
        accelerator='gpu',
        precision=32,
        max_epochs=200,
        log_every_n_steps=1,
        enable_progress_bar=True,
        callbacks=callbacks
    )

    trainer.fit(model=model, datamodule=datamodule)
    
    
        
    
