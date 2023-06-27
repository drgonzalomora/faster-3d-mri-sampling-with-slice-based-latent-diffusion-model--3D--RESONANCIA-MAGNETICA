import torch
import pytorch_lightning as pl
from .unet import UNetModel
from .ema import LitEma
from contextlib import contextmanager

class DDPM(pl.LightningModule):
    def __init__(
        self,
        unet_config,
        diffusion,
        sampler,
        use_ema     = True,
        clamp       = True,
        lr          = 5e-05,
        weight_decay = 0.0,
        **kwargs
        ) -> None:
        super().__init__()
        self.model = UNetModel(**unet_config)
        self.diffusion = diffusion
        self.sampler = sampler
        self.use_ema = use_ema
        self.clamp = clamp

        if self.use_ema:
            self.ema = LitEma(self.model, decay=0.9999)
            print(f"Keeping EMAs of {len(list(self.ema.buffers()))}.")

        self.save_hyperparameters(ignore=['diffusion', 'sampler'])

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.ema.store(self.model.parameters())
            self.ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.ema(self.model)

    def training_step(self, batch, batch_idx):
        x_i = batch[0]
        x_i = x_i.to(dtype=self.model.precision)

        # forward step
        times, weights = self.sampler.sample(device=self.device)
        
        eps = torch.randn_like(x_i)
        loss = self.diffusion.forward_step(self.model, x_i, times, noise=eps, weights=weights)
        
        # logging loss
        self.log('mse_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay, 
            betas=(0.5, 0.9)
        )
        
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.trainer.max_steps, eta_min=1e-9, last_epoch=-1
            ),
            'interval': 'step',
            'frequency': 1
        }
        
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    def sample_img(
        self, 
        n_samples   = 1, 
        noise       = None, 
        ddim        = False, 
        eta         = 0.0, 
        clamp       = True, 
        intermediate = False
    ):
        if noise is None:
            noise = torch.randn(
                n_samples, 
                self.model.in_channels,
                self.model.image_size[0], 
                self.model.image_size[1]
            ).to(self.device, dtype=torch.float32)
        
        samples = self.diffusion.sample(
            self.model, noise, ddim=ddim, eta=eta, clamp=clamp, intermediate=intermediate
        )

        return samples


