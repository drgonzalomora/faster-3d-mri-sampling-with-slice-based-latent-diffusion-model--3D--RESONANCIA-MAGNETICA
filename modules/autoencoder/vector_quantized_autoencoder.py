from typing import Any, Optional
import numpy as np
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
import pytorch_lightning as pl

from .modules import (
    TimePositionalEmbedding, EncodingBlock, DecodingBlock,
    ResidualBlock, SelfAttention
)

from ..loss.vector_quantizer import VectorQuantizer
from ..loss.lpips import VQLPIPSWithDiscriminator

class Encoder(nn.Module):
    def __init__(
        self, in_channels, z_channels=4, pemb_dim=None, num_channels=128, channels_mult=[1, 2, 4, 4], 
        num_res_blocks=2, attn=None
        ) -> None:
        super().__init__()
        if attn is not None:
            assert channels_mult.__len__() == attn.__len__(), 'channels_mult and attn must have the same length'
            self.attn = attn
        else:
            self.attn = [False] * channels_mult.__len__()

        self.z_channels = z_channels
        self.channels_mult = [1, *channels_mult]
        self.attn = attn
        
        # architecture modules
        self.in_conv = nn.Conv2d(in_channels, num_channels, kernel_size=3, padding='same')
        self.enocoder = nn.ModuleList([
            EncodingBlock(
                in_channels=num_channels * self.channels_mult[idx],
                out_channels=num_channels * self.channels_mult[idx + 1],
                temb_dim=pemb_dim,
                num_blocks=num_res_blocks,
                attn=self.attn[idx],
                downsample=True if idx != self.channels_mult.__len__() - 2 else False
            ) for idx in range(self.channels_mult.__len__() - 1)
        ])
        bottleneck_channels = num_channels * self.channels_mult[-1]
        seeneck_channels = num_channels * self.channels_mult[-1]
        self.bottleneck_res_a = ResidualBlock(in_channels=bottleneck_channels, out_channels=bottleneck_channels, temb_dim=pemb_dim, groups=8)
        self.bottleneck_sa = SelfAttention(in_channels=bottleneck_channels, num_heads=8, head_dim=32, groups=8)
        self.bottleneck_res_b = ResidualBlock(in_channels=bottleneck_channels, out_channels=bottleneck_channels, temb_dim=pemb_dim, groups=8)
        self.out_conv = nn.Sequential(
            nn.GroupNorm(num_groups=8, num_channels=bottleneck_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels=bottleneck_channels, out_channels=self.z_channels, kernel_size=3, padding=1)
        )
    
    def forward(self, x, pemb=None):
        x = self.in_conv(x)
        for encoder in self.enocoder:
            x = encoder(x, pemb)
        x = self.bottleneck_res_a(x, pemb)
        x = self.bottleneck_sa(x)
        x = self.bottleneck_res_b(x, pemb)
        x = self.out_conv(x)
        return x
    
class Decoder(nn.Module):
    def __init__(
        self, out_channels, z_channels, pemb_dim=None, num_channels=128, channels_mult=[1, 2, 4, 4],
        num_res_blocks=2, attn=None
        ) -> None:
        super().__init__()
        if attn is not None:
            assert channels_mult.__len__() == attn.__len__(), 'channels_mult and attn must have the same length'
            self.attn = list(reversed(attn))
        else: 
            self.attn = [False] * channels_mult.__len__()

        self.channels_mult = list(reversed([1, *channels_mult]))
        self.z_channels = z_channels
        
        # architecture modules
        bottleneck_channels = num_channels * self.channels_mult[0]
        self.in_conv = nn.Conv2d(self.z_channels, bottleneck_channels, kernel_size=3, padding='same')
        self.bottleneck_res_a = ResidualBlock(in_channels=bottleneck_channels, out_channels=bottleneck_channels, temb_dim=pemb_dim, groups=8)
        self.bottleneck_sa = SelfAttention(in_channels=bottleneck_channels, num_heads=8, head_dim=32, groups=8)
        self.bottleneck_res_b = ResidualBlock(in_channels=bottleneck_channels, out_channels=bottleneck_channels, temb_dim=pemb_dim, groups=8)

        self.decoder = nn.ModuleList([
            DecodingBlock(
                in_channels=num_channels * self.channels_mult[idx],
                out_channels=num_channels * self.channels_mult[idx + 1],
                temb_dim=pemb_dim,
                num_blocks=num_res_blocks,
                attn=self.attn[idx],
                upsample=True if idx != 0 else False
            ) for idx in range(self.channels_mult.__len__() - 1)
        ])
        
        self.out_conv = nn.Sequential(
            nn.GroupNorm(num_groups=8, num_channels=num_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels=num_channels, out_channels=out_channels, kernel_size=3, padding=1)
        )
        
    def forward(self, x, pemb=None):
        x = self.in_conv(x)
        x = self.bottleneck_res_a(x, pemb)
        x = self.bottleneck_sa(x)
        x = self.bottleneck_res_b(x, pemb)
        for decoder in self.decoder:
            x = decoder(x, pemb)
        x = self.out_conv(x)
        return x

class VQAutoencoder(pl.LightningModule):
    def __init__(self,
        input_shape=[2, 128, 128],
        n_embed=8192, 
        embed_dim=2,
        z_channels=4, 
        z_double=False, 
        pemb_dim=128, 
        T=64,
        num_channels=128, 
        channels_mult=[1, 2, 4], 
        num_res_blocks=2, 
        attn=[],
        learning_rate=1e-5,
        lr_d_factor=1.,
        **kwargs
    ) -> None:
        super().__init__()
        if attn.__len__() > 0:
            assert channels_mult.__len__() == attn.__len__(), 'channels_mult and attn must have the same length'
            self.attn = attn
        else:
            self.attn = [False] * channels_mult.__len__()

        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.z_channels = z_channels if not z_double else z_channels * 2
        self.learning_rate = learning_rate
        self.lr_d_factor = lr_d_factor
        in_channels = out_channels = input_shape[0]

        # architecture modules
        self.positional_encoder = nn.Sequential(
            TimePositionalEmbedding(dimension=pemb_dim, T=T, device='cuda'),
            nn.Linear(pemb_dim, 128 * 4),
            nn.GELU(),
            nn.Linear(128 * 4, pemb_dim)
        )
        self.encoders = nn.ModuleList([
            Encoder(1, z_channels, pemb_dim, num_channels, channels_mult, num_res_blocks, self.attn) for _ in range(in_channels)
        ])
        decoder_in_channels = in_channels * z_channels
        vq_embed_dim = self.embed_dim * in_channels
        self.decoder = Decoder(out_channels, decoder_in_channels, pemb_dim, num_channels, channels_mult, num_res_blocks, self.attn)
        self.quantizer = VectorQuantizer(self.n_embed, vq_embed_dim, beta=0.25, remap=None)
        self.quant_conv = nn.Conv2d(decoder_in_channels, vq_embed_dim, kernel_size=1)
        self.post_quant_conv = nn.Conv2d(vq_embed_dim, decoder_in_channels, kernel_size=1)

        # loss functions
        self.loss = VQLPIPSWithDiscriminator(**kwargs['loss'])

        # TODO: Add EMA
        
        # pytorch lightining states
        self.automatic_optimization = False
        self.save_hyperparameters()

    def encode(self, x, pemb):
        z_i = []
        # forwarding each channel through its own encoder
        for c_i, encoder in enumerate(self.encoders):
            x_i = x[:, c_i, None]
            z_i.append(encoder(x_i, pemb))

        # concatenating the channels
        z = torch.cat(z_i, dim=1)
        z = self.quant_conv(z)
        z_q, qloss, info = self.quantizer(z)
        return z_q, z_i, qloss, info
    
    def encode_pre_quantization(self, x, pemb):
        z_i = []
        # forwarding each channel through its own encoder
        for c_i, encoder in enumerate(self.encoders):
            z_i.append(encoder(x[:, c_i, None], pemb))

        # concatenating the channels
        z = torch.cat(z_i, dim=1)
        z = self.quant_conv(z)
        return z, z_i
    
    def decode(self, z_q, pemb):
        z_q = self.post_quant_conv(z_q)
        # TODO: Cross-attention avec mask ici
        x = self.decoder(z_q, pemb)
        return x
    
    def decode_code(self, code_b, pemb):
        z_q = self.quantizer.embedding(code_b)
        x = self.decode(z_q, pemb)
        return torch.tanh(x)
    
    def decode_pre_quantization(self, z, pemb):
        z_q, qloss, info = self.quantizer(z)
        x = self.decode(z_q, pemb)
        return torch.tanh(x), qloss, info
    
    def encode_position(self, position):
        return self.positional_encoder(position)
    
    def forward(self, x, position, return_indices=False):
        pemb = self.positional_encoder(position)
        z_q, z_i, qloss, (_, _, indices) = self.encode(x, pemb)
        x = self.decode(z_q, pemb)
        if return_indices:
            return torch.tanh(x), z_i, qloss, indices
        return torch.tanh(x), z_i, qloss

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        pass # TODO: EMA

    def training_step(self, batch, batch_idx):
        # optimizers & schedulers
        ae_opt, disc_opt = self.optimizers()
        ae_scheduler, disc_scheduler = self.lr_schedulers()

        x, pos = batch
        x, pos = x.type(torch.float16), pos.type(torch.long)
        
        x_hat, z_i, qloss, _ = self.forward(x, pos, return_indices=True)

        ########################
        # Optimize Autoencoder #
        ########################
        ae_loss, ae_log = self.loss.autoencoder_loss(qloss, x, x_hat, z_i, self.global_step, last_layer=self.decoder.out_conv[-1].weight)
        ae_opt.zero_grad(set_to_none=True)
        self.manual_backward(ae_loss)
        ae_opt.step()
        ae_scheduler.step()

        ##########################
        # Optimize Discriminator #
        ##########################
        disc_loss, disc_log = self.loss.discriminator_loss(x, x_hat, self.global_step)
        disc_opt.zero_grad(set_to_none=True)
        self.manual_backward(disc_loss)
        disc_opt.step()
        disc_scheduler.step()

        # logging
        self.log_dict(ae_log, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict(disc_log, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        
    def validation_step(self, batch, batch_idx):
        x, pos = batch
        x, pos = x.type(torch.float16), pos.type(torch.long)
        
        x_hat, z_i, qloss, _ = self.forward(x, pos, return_indices=True)
        ae_loss, ae_log = self.loss.autoencoder_loss(
            qloss, x, x_hat, z_i, self.global_step, last_layer=self.decoder.out_conv[-1].weight, split='val'
        )

        self.log_dict(ae_log, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    
    def configure_optimizers(self):
        ae_opt = torch.optim.AdamW(list(self.encoders.parameters()) + 
                                   list(self.decoder.parameters()) + 
                                   list(self.positional_encoder.parameters()) +
                                   list(self.quant_conv.parameters()) +
                                   list(self.post_quant_conv.parameters()) +
                                   list(self.quantizer.parameters()), 
                                   lr=self.learning_rate, weight_decay=1e-6, betas=(0.5, 0.9))
        disc_opt = torch.optim.AdamW(list(self.loss.discriminator.parameters()), 
                                     lr=self.learning_rate * self.lr_d_factor, weight_decay=1e-6, betas=(0.5, 0.9))
        
        schedulers = [
            {
                'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(
                    ae_opt, T_max=self.trainer.max_steps, eta_min=1e-9, last_epoch=-1
                ),
                'interval': 'step',
                'frequency': 1
            },
            {
                'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(
                    disc_opt, T_max=self.trainer.max_steps, eta_min=1e-8, last_epoch=-1
                ),
                'interval': 'step',
                'frequency': 1
            }
        ]

        return [ae_opt, disc_opt], schedulers
    



    
    
