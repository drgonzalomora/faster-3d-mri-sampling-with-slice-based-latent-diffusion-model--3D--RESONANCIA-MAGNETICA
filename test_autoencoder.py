import os
import torch
from modules.autoencoder.vector_quantized_autoencoder import VQAutoencoder
from modules.preprocessing import BRATSDataModule
from omegaconf import OmegaConf

from tqdm import tqdm
import torchvision.utils as vutils
from PIL import Image

if __name__ == "__main__":
    # loading config file
    CONFIG_PATH = './config.yaml'
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError('Config file not found')
    
    cfg = OmegaConf.load(CONFIG_PATH)
    assert cfg.target in ['first-stage-training', 'diffusion-training', 'inference']
    
    print('loading data...')
    dm = BRATSDataModule(**cfg.data.first_stage)
    dm.setup()
    test_dataloader = dm.test_dataloader()
    
    # loading model
    ae = VQAutoencoder.load_from_checkpoint('./checkpoints/autoencoder-VQAutoencoder.ckpt')

    # inference
    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_dataloader)):
            x, pos = batch
            recon_x, qloss = ae(x, timestep=pos, return_indices=False)
            img_grid = vutils.make_grid(torch.cat([x, recon_x], dim=0), nrow=x.shape[0])
            img = Image.fromarray(img_grid.add(1).mul(127.5).permute(1, 2, 0).cpu().numpy().astype('uint8'))
            img.save(f'samples/sample_{i}.png')
    