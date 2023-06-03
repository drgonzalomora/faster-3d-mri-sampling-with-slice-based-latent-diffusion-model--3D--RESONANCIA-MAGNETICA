import torch
import pytorch_lightning as pl
import wandb

from .autoencoder.gaussian_autoencoder import GaussianAutoencoder
from .autoencoder.vector_quantized_autoencoder import VQAutoencoder

class ImageReconstructionLogger(pl.Callback):
    def __init__(self,
        modalities=['t1', 't1ce', 't2', 'flair'], 
        n_samples=5,
        **kwargs
    ) -> None:
        super().__init__()
        self.n_samples = n_samples
        self.modalities = modalities

    def on_train_epoch_end(self, trainer, pl_module):
        # sample images
        pl_module.eval()
        with torch.no_grad():
            x, pos = next(iter(trainer.train_dataloader))
            x, pos = x.to(pl_module.device, torch.float32), pos.to(pl_module.device, torch.long)

            x, pos = x[:self.n_samples], pos[:self.n_samples]
            x_hat = pl_module(x, pos)[0]

            originals = torch.cat([
                torch.hstack([img for img in x[:, idx, ...]])
                for idx in range(self.modalities.__len__())
            ], dim=0)
            
            reconstructed = torch.cat([
                torch.hstack([img for img in x_hat[:, idx, ...]])
                for idx in range(self.modalities.__len__())
            ], dim=0)
            
            img = torch.cat([originals, reconstructed], dim=0)
            
            wandb.log({
                'Reconstruction examples': wandb.Image(
                    img.detach().cpu().numpy(), 
                    caption='{} - {} (Top are originals)'.format(self.modalities, trainer.current_epoch)
                )
            })
            

class ImageGenerationLogger(pl.Callback):
    def __init__(self,
        autoencoder,
        n_samples=5,
        every_n_epochs=50,
        **kwargs
    ) -> None:
        super().__init__()
        self.autoencoder = autoencoder
        self.n_samples = n_samples
        self.every_n_epochs = every_n_epochs
        
    def on_train_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.every_n_epochs == 0:
            # sample images
            pl_module.eval()
            with torch.no_grad():
                noise = torch.randn(
                    1,
                    pl_module.in_channels,
                    *pl_module.image_size
                ).to(pl_module.device, torch.float32)
                
                # diffusion the noise
                x_hat = pl_module.diffusion.sample(pl_module, noise, ddim=False, clamp=True, verbose=True)
                x_hat = x_hat.permute(0, 2, 1, 3, 4).squeeze(0) # slices as batch, 3D to 2D
                
                x_hat = x_hat[::int(x_hat.shape[0] / 10), ...] # => will be of shape (10, **image_size)
                positions = torch.arange(0, x_hat.shape[0]).to(pl_module.device, torch.long)[::int(x_hat.shape[0] / 10)]
                
                x_hat = x_hat.to(self.autoencoder.device, torch.float32)
                positions = positions.to(self.autoencoder.device, torch.long)
                
                # decoding
                x_hat = x_hat.type(torch.float32) # TODO: check diffusion output dtype
                pemb = self.autoencoder.encode_position(positions)
                if isinstance(self.autoencoder, GaussianAutoencoder):
                    x_hat = self.autoencoder.decode(x_hat, pemb)
                    x_hat = torch.tanh(x_hat)
                elif isinstance(self.autoencoder, VQAutoencoder):
                    x_hat, _, _ = self.autoencoder.decode_pre_quantization(x_hat, pemb)
                else:
                    raise NotImplementedError('Unknown autoencoder type')
                
                # at this point, the number of channels should be the number of modalities
                # 10 x NUM_MODALITIES x H x W
                img = torch.cat([
                    torch.hstack([img for img in x_hat[:, idx, ...]])
                    for idx in range(x_hat.shape[1])
                ], dim=0)

                wandb.log({
                    'Generated examples': wandb.Image(
                        img.detach().cpu().numpy(), 
                        caption='Generated examples (epoch {})'.format(trainer.current_epoch)
                    )
                })