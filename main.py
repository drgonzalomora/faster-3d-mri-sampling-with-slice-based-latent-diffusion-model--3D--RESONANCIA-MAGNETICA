import numpy as np
import torch
import pytorch_lightning as pl
import os
import sys
import glob
from omegaconf import OmegaConf
from pytorch_lightning.loggers import wandb as wandb_logger
from pytorch_lightning.callbacks import ModelCheckpoint

from modules.preprocessing import BRATSDataModule, BRATSLatentsDataModule
from modules.autoencoder.gaussian_autoencoder import GaussianAutoencoder
from modules.autoencoder.vector_quantized_autoencoder import VQAutoencoder
from modules.loggers import ImageReconstructionLogger, ImageGenerationLogger
from modules.unet import UNetModel
from modules.diffusion import Diffusion, SimpleDiffusion
from modules.sampler import ScheduleSampler

os.environ['WANDB_API_KEY'] = 'bdc8857f9d6f7010cff35bcdc0ae9413e05c75e1'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        project='guided-latent-diffusion', 
        name='{}-{}'.format(
            cfg.target,
            cfg.autoencoder.target
        ),
        # id='24hyhi7b',
        # resume="must"
    )
    
    callbacks = []
    if cfg.target == 'first-stage-training':
        # data module
        datamodule = BRATSDataModule(**cfg.data.first_stage)
        
        # autoencoder
        if cfg.autoencoder.target == 'VQAutoencoder':
            model = VQAutoencoder(**cfg.autoencoder, **cfg.autoencoder.loss)
        elif cfg.autoencoder.target == 'GaussianAutoencoder':
            model = GaussianAutoencoder(**cfg.autoencoder, **cfg.autoencoder.loss)
        else:
            raise ValueError('Unknown autoencoder target')
        
        callbacks.append(
            ImageReconstructionLogger(
                modalities=['FLAIR', 'SEG'],
                n_samples=5
            )
        )
    
    elif cfg.target == 'diffusion-training':
        # loading autoencoder
        try: # autoencoder
            target_class = getattr(sys.modules[__name__], cfg.autoencoder.target)
        except:
            raise AttributeError('Unknown autoencoder target')
        
        # load autoencoder weights/hyperparameters
        ckpt_list = glob.glob('./checkpoints/autoencoder-' + cfg.autoencoder.target + '*.ckpt')
        autoencoder = target_class.load_from_checkpoint(ckpt_list[-1]) # latest weights
        print('Using autoencoder: ', type(autoencoder).__name__)
        print('Loaded autoencoder weights from: ', ckpt_list[-1])
        
        # data module
        datamodule = BRATSLatentsDataModule(autoencoder=autoencoder, **cfg.data.diffusion)
        
        sampler = ScheduleSampler(
            T=cfg.diffusion.T,
            batch_size=cfg.data.diffusion.batch_size,
            sampler=cfg.diffusion.schedule_sampler,
            memory_span=cfg.diffusion.loss_memory_span,
            device=device
        )

        diffusion = Diffusion(**cfg.diffusion)
        # diffusion = SimpleDiffusion(
        #     noise_shape=[4, 256, 256],
        #     T=cfg.diffusion.T,
        #     beta_schedule='cosine'
        # )
        
        if cfg.unet.use_checkpoint:
            print('Resuming training from checkpoint ... ')
            # load unet weights/hyperparameters
            ckpt_list = glob.glob('./checkpoints/diffusion-' + cfg.autoencoder.target + '*.ckpt')
            model = UNetModel.load_from_checkpoint(
                ckpt_list[-1], # TODO: Fix checkpoint missing hyperparams
                **cfg.unet,
                diffusion=diffusion,
                sampler=sampler
            )
            print('Loaded UNet weights from: ', ckpt_list[-1])
        else:
            print('Creating a fresh UNet model ...')
            model = UNetModel(diffusion=diffusion, sampler=sampler, **cfg.unet)
            
        callbacks.append(
            ImageGenerationLogger(
                autoencoder,
                n_samples=1,
                to_2d=True,
                every_n_epochs=25
            )
        )
    
    elif cfg.target == 'inference':
        pass
    
    # callbacks
    callbacks.append(
        ModelCheckpoint(
            **cfg.callbacks.checkpoint,
            filename='{}-{}'.format(
                'diffusion' if cfg.target == 'diffusion-training' else 'autoencoder',
                cfg.autoencoder.target
            )
        )
    )
    
    # training
    trainer = pl.Trainer(
        logger=logger,
        # strategy="ddp",
        # devices=4,
        # num_nodes=1,
        accelerator='gpu',
        precision=32,
        max_epochs=200,
        log_every_n_steps=1,
        enable_progress_bar=True,
        callbacks=callbacks
    )

    trainer.fit(model=model, datamodule=datamodule)
    
    
        
    
