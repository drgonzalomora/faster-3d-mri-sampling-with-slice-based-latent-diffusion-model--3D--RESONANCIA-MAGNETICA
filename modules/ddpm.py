import torch
import pytorch_lightning as pl
from .unet import UNetModel



class DDPM(pl.LightningModule):
    def __init__(
        self,
        input_shape,
        timesteps,
        beta_schedule   = 'linear',
        loss_type       = 'l1',
        use_ema         = True,
        clip            = True,
        **kwargs
        ) -> None:
        super().__init__()
        
        self.input_shape = input_shape
        self.clip = clip
        self.in_channels = input_shape[0]
        self.model = UNetModel(
            
        )
        