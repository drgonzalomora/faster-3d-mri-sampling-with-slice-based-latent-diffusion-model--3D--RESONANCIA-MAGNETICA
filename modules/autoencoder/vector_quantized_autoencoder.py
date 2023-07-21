import torch
import torch.nn as nn
import pytorch_lightning as pl
from contextlib import contextmanager

from .base_autoencoder import (
    TimePositionalEmbedding, EncodingBlock, DecodingBlock,
    ResidualBlock, SelfAttention
)

from ..loss.vector_quantizer import VectorQuantizer
from ..loss.lpips import VQLPIPSWithDiscriminator
from ..ema import LitEma

class Encoder(nn.Module):
    def __init__(
        self,
        in_channels,
        num_channels        = 128,
        channels_mult       = [1, 2, 4, 8],
        num_res_blocks      = 2,
        attn                = None, 
        temb_dim            = None,
        dropout             = 0.0,
        z_channels          = 2,
        double_z            = False,
        **kwargs
    ) -> None:
        super().__init__()

        # architecture components
        self.in_conv = nn.Conv2d(in_channels, num_channels, kernel_size=3, padding='same') # equivalent to padding=1, stride=1
        self.channels_mult = (1,) + tuple(channels_mult)
        self.attn = attn if (attn != None and attn != False and attn != []) else [False] * self.channels_mult.__len__()

        self.enocoder = nn.ModuleList([
            EncodingBlock(
                in_channels=num_channels * self.channels_mult[idx],
                out_channels=num_channels * self.channels_mult[idx + 1],
                temb_dim=temb_dim,
                num_blocks=num_res_blocks,
                attn=self.attn[idx],
                downsample=True if idx != self.channels_mult.__len__() - 2 else False,
                dropout=dropout
            ) for idx in range(self.channels_mult.__len__() - 1)
        ])

        # bottleneck        
        bottleneck_channels = num_channels * self.channels_mult[-1]
        self.bottleneck_res_a = ResidualBlock(
            in_channels=bottleneck_channels, 
            out_channels=bottleneck_channels, 
            temb_dim=temb_dim, 
            groups=8,
            dropout=dropout
        )
        self.bottleneck_sa = SelfAttention(
            in_channels=bottleneck_channels, 
            num_heads=8, 
            head_dim=32, 
            groups=32
        )
        self.bottleneck_res_b = ResidualBlock(
            in_channels=bottleneck_channels, 
            out_channels=bottleneck_channels, 
            temb_dim=temb_dim, 
            groups=8,
            dropout=dropout
        )
        
        self.out_conv = nn.Sequential(
            nn.GroupNorm(num_groups=8, num_channels=bottleneck_channels),
            nn.SiLU(),
            nn.Conv2d(
                in_channels=bottleneck_channels, 
                out_channels=z_channels if not double_z else z_channels * 2, 
                kernel_size=3, padding=1, stride=1
            )
        )
    
    def forward(self, x, temb=None):
        x = self.in_conv(x)

        # encoding
        for encoder in self.enocoder:
            x = encoder(x, temb)

        # bottleneck
        x = self.bottleneck_res_a(x, temb)
        x = self.bottleneck_sa(x)
        x = self.bottleneck_res_b(x, temb)
        
        # out
        x = self.out_conv(x)
        return x
    
class Decoder(nn.Module):
    def __init__(
        self,
        out_channels,
        num_channels        = 128,
        channels_mult       = [1, 2, 4, 8],
        num_res_blocks      = 2,
        attn                = None, 
        temb_dim            = None,
        dropout             = 0.0,
        z_channels          = 2,
        double_z            = False,
        tanh                = False,
        **kwargs
    ) -> None:
        super().__init__()

        self.attn = tuple(reversed(attn)) \
                    if (attn != None and attn != False and attn != []) else [False] * channels_mult.__len__()

        self.channels_mult = tuple(reversed(channels_mult)) + (1,)
        self.z_channels = z_channels if not double_z else z_channels * 2
        
        # architecture components
        bottleneck_channels = num_channels * self.channels_mult[0]
        self.in_conv = nn.Conv2d(self.z_channels, bottleneck_channels, kernel_size=3, padding='same')

        # bottleneck
        self.bottleneck_res_a = ResidualBlock(
            in_channels=bottleneck_channels, 
            out_channels=bottleneck_channels, 
            temb_dim=temb_dim, 
            groups=32,
            dropout=dropout
        )
        
        self.bottleneck_sa = SelfAttention(
            in_channels=bottleneck_channels, 
            num_heads=8, 
            head_dim=32, 
            groups=32
        )

        self.bottleneck_res_b = ResidualBlock(
            in_channels=bottleneck_channels, 
            out_channels=bottleneck_channels, 
            temb_dim=temb_dim, 
            groups=32,
            dropout=dropout
        )

        self.decoder = nn.ModuleList([
            DecodingBlock(
                in_channels=num_channels * self.channels_mult[idx],
                out_channels=num_channels * self.channels_mult[idx + 1],
                temb_dim=temb_dim,
                num_blocks=num_res_blocks,
                attn=self.attn[idx],
                upsample=True if idx != 0 else False,
                dropout=dropout
            ) for idx in range(self.channels_mult.__len__() - 1)
        ])
        
        self.out_conv = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=num_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels=num_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.Tanh() if tanh else nn.Identity()
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
    def __init__(
        self,
        input_shape,
        n_embed, 
        embed_dim,
        num_channels    = 128, 
        channels_mult   = [1, 2, 4], 
        num_res_blocks  = 2, 
        attn            = None,
        temb_dim        = 128, 
        max_period      = 64,
        dropout         = 0.0,
        z_channels      = 2, 
        z_double        = False, 
        tanh            = False,
        use_ema         = False,
        learning_rate   = 1e-5,
        **kwargs
    ) -> None:
        super().__init__()
        in_channels = out_channels = input_shape[0]
        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.z_channels = z_channels if not z_double else z_channels * 2
        self.attn = attn if (attn != None and attn != False and attn != []) else [False] * channels_mult.__len__()
        self.learning_rate = learning_rate

        # perceptual quantizer loss function
        self.loss = VQLPIPSWithDiscriminator(**kwargs['loss'])

        # architecture components
        self.temb_dim = temb_dim
        if self.temb_dim != None:
            self.positional_encoder = nn.Sequential(
                TimePositionalEmbedding(dimension=temb_dim, T=max_period, device='cuda'),
                nn.Linear(temb_dim, temb_dim * 4),
                nn.GELU(),
                nn.Linear(temb_dim * 4, temb_dim)
            )

        self.quant_conv = nn.Conv2d(z_channels, embed_dim, kernel_size=1)
        self.post_quant_conv = nn.Conv2d(embed_dim, z_channels, kernel_size=1)

        self.encoder = Encoder(
            in_channels, num_channels, channels_mult, num_res_blocks, attn, temb_dim, dropout, z_channels, z_double
        )
        self.decoder = Decoder(
            out_channels, num_channels, channels_mult, num_res_blocks, attn, temb_dim, dropout, z_channels, z_double, tanh
        )

        self.quantizer = VectorQuantizer(self.n_embed, embed_dim, beta=0.25)
        
        # ema
        self.use_ema = use_ema
        if use_ema:
            self.ema = LitEma(self)
            print(f"Keeping EMAs of {len(list(self.ema.buffers()))}.")
        
        # pytorch lightining states
        self.automatic_optimization = False
        self.save_hyperparameters()

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.ema.store(self.parameters())
            self.ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.ema(self)

    def encode(self, x, temb=None):
        # z_i = []
        # # forwarding each channel through its own encoder
        # for c_i, encoder in enumerate(self.encoders):
        #     x_i = x[:, c_i, None]
        #     z_i.append(encoder(x_i, pemb))
        z = self.encoder(x, temb)

        # concatenating the channels
        # z = torch.cat(z_i, dim=1)

        z = self.quant_conv(z)
        z_q, qloss, info = self.quantizer(z)
        return z_q, qloss, info
    
    def encode_pre_quantization(self, x, temb=None):
        # z_i = []
        # # forwarding each channel through its own encoder
        # for c_i, encoder in enumerate(self.encoders):
        #     z_i.append(encoder(x[:, c_i, None], pemb))

        # concatenating the channels
        # z = torch.cat(z_i, dim=1)
        z = self.encoder(x, temb)
        z = self.quant_conv(z)
        return z
    
    def decode(self, z_q, temb=None):
        z_q = self.post_quant_conv(z_q)
        # TODO: Cross-attention avec mask ici
        x = self.decoder(z_q, temb)
        return x
    
    def decode_code(self, code_b, temb=None):
        z_q = self.quantizer.embedding(code_b)
        x = self.decode(z_q, temb)
        return x
    
    def decode_pre_quantization(self, z, pemb):
        z_q, qloss, info = self.quantizer(z)
        x = self.decode(z_q, pemb)
        return x, qloss, info
    
    def encode_timestep(self, timestep):
        return self.positional_encoder(timestep)
    
    def forward(self, x, timestep=None, return_indices=False):
        if timestep is not None and self.temb_dim is not None:
            temb = self.positional_encoder(timestep)
        else:
            temb = None

        z_q, qloss, (_, _, indices) = self.encode(x, temb)
        x = self.decode(z_q, temb)
        if return_indices:
            return x, qloss, indices
        return x, qloss

    def training_step(self, batch, batch_idx):
        # optimizers & schedulers
        ae_opt, disc_opt = self.optimizers()
        ae_scheduler, disc_scheduler = self.lr_schedulers()

        x, pos = batch
        x, pos = x.type(torch.float32), pos.type(torch.long)
        
        x_hat, qloss, _ = self.forward(x, pos, return_indices=True)

        ########################
        # Optimize Autoencoder #
        ########################
        ae_loss, ae_log = self.loss.autoencoder_loss(qloss, x, x_hat, self.global_step, last_layer=self.decoder.out_conv[-1].weight)
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
        log_dict = self._validation_step(batch, batch_idx)
        with self.ema_scope():
            log_dict_ema = self._validation_step(batch, batch_idx, suffix="_ema")
        return log_dict
        
    def _validation_step(self, batch, batch_idx, suffix=""):
        x, pos = batch
        x, pos = x.type(torch.float32), pos.type(torch.long)
        
        x_hat, z_i, qloss, _ = self.forward(x, pos, return_indices=True)
        _, ae_log = self.loss.autoencoder_loss(
            qloss, x, x_hat, z_i, self.global_step, last_layer=self.decoder.out_conv[-1].weight, split=suffix
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
    



    
    
