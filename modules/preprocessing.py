import numpy as np
import torch
import pytorch_lightning as pl
from nibabel import load
from nibabel.processing import resample_to_output
from tqdm import tqdm
import os

from .autoencoder.gaussian_autoencoder import GaussianAutoencoder
from .autoencoder.vector_quantized_autoencoder import VQAutoencoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class IdentityDataset(torch.utils.data.Dataset):
    def __init__(self, *data):
        self.data = data

    def __len__(self):
        return self.data[-1].__len__()

    def __getitem__(self, index):
        return [d[index] for d in self.data]

class BRATSDataModule(pl.LightningDataModule):
    def __init__(self,
        target_shape    = (64, 128, 128),
        n_samples       = 500,
        modalities      = ['t1', 't1ce', 't2', 'flair', 'seg'],
        binarize        = True,
        npy_path        = '../data/brats_preprocessed.npy',
        root_path       = '../../common_data/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021',
        batch_size      = 32,
        shuffle         = True,
        num_workers     = 6,
        **kwargs
    ) -> None:
        assert all([m in ['t1', 't1ce', 't2', 'flair', 'seg'] for m in modalities]), 'Invalid modality!'
        
        super().__init__()
        self.num_modalities = len(modalities)
        self.prepare_data_per_node = True

        # just for a faster access
        self.save_hyperparameters()

    def prepare_data(self) -> None:
        if not os.path.exists(self.hparams.npy_path):
            print('Loading dataset from NiFTI files...')
            placeholder = np.zeros(shape=(
                self.hparams.n_samples, 
                self.num_modalities, 
                self.hparams.target_shape[1], 
                self.hparams.target_shape[2], 
                self.hparams.target_shape[0]
            ))

            for idx, instance in enumerate(tqdm(os.listdir(self.hparams.root_path)[: self.hparams.n_samples], position=0, leave=True)):
                # loading models
                volumes = {}
                for _, m in enumerate(self.hparams.modalities):
                    volumes[m] = load(os.path.join(self.hparams.root_path, instance, instance + f'_{m}.nii.gz'))

                # Compute the scaling factors (output will not be exactly the same as defined in OUTPUT_SHAPE)
                orig_shape = volumes[self.hparams.modalities[0]].shape
                scale_factor = (orig_shape[0] / self.hparams.target_shape[1], # height
                                orig_shape[1] / self.hparams.target_shape[2], # width
                                orig_shape[2] / self.hparams.target_shape[0]) # depth

                # Resample the image using trilinear interpolation
                # Drop the last extra rows/columns/slices to get the exact desired output size
                for _, m in enumerate(self.hparams.modalities):
                    volumes[m] = resample_to_output(volumes[m], voxel_sizes=scale_factor, order=1).get_fdata()
                    volumes[m] = volumes[m][:self.hparams.target_shape[1], :self.hparams.target_shape[2], :self.hparams.target_shape[0]]

                # binarizing the mask (for simplicity), you can comment out this to keep all labels
                if self.hparams.binarize and 'seg' in self.hparams.modalities:
                    volumes['seg'] = (volumes['seg'] > 0).astype(np.float32)

                # saving models
                for idx_m, m in enumerate(self.hparams.modalities):
                    placeholder[idx, idx_m, :, :] = volumes[m]

            print('Saving dataset as npy file...')    
            # saving the dataset as a npy file
            np.save(self.hparams.npy_path, placeholder)
                
            print('Saved!')
        else:
            print('Dataset already exists at {}'.format(self.hparams.npy_path))
        
    def setup(self, stage='fit'):
        assert os.path.exists(self.hparams.npy_path), 'npy data file does not exist!'
        
        print('Loading dataset from npy file...')
        self.data = torch.from_numpy(np.load(self.hparams.npy_path))
        self.data = self.data[:self.hparams.n_samples]
        
        # normalize the data [-1, 1]
        norm = lambda data: data * 2 / data.max() - 1
        for m in range(self.num_modalities):
            for idx in range(self.hparams.n_samples):
                self.data[idx, m] = norm(self.data[idx, m]).type(torch.float32)
        self.data.clamp(-1, 1)

        self.data = self.data.permute(0, 4, 1, 2, 3) # depth first
            
        # if switching to 2D for autoencoder training
        D, W, H = self.hparams.target_shape
        self.data = self.data.reshape(self.hparams.n_samples * D, -1, W, H)

        # keeping track on slice positions for positional embedding
        self.slice_positions = torch.arange(D)[None, :].repeat(self.hparams.n_samples, 1)
        self.slice_positions = self.slice_positions.flatten()
        
        train_size = int(0.85 * self.data.shape[0])
        
        self.train_x = self.data[:train_size]
        self.train_pos = self.slice_positions[:train_size]
        self.test_x = self.data[train_size:]
        self.test_pos = self.slice_positions[train_size:]

        print('Train shape:', self.train_x.shape) 
        print('Test shape:', self.test_x.shape)
        print('Train slice positions shape:', self.train_pos.shape)
        print('Test slice positions shape:', self.test_pos.shape)
        print('Min: {}, Max: {}'.format(self.data.min(), self.data.max()))
        
        self.train_dataset = IdentityDataset(self.train_x, self.train_pos)
        self.test_dataset = IdentityDataset(self.test_x, self.test_pos)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, 
            batch_size=self.hparams.batch_size, 
            shuffle=self.hparams.shuffle, 
            num_workers=self.hparams.num_workers, 
            pin_memory=True
        )
        
    # def val_dataloader(self):
    #     return torch.utils.data.DataLoader(
    #         self.test_dataset, 
    #         batch_size=self.hparams.batch_size, 
    #         shuffle=False, 
    #         num_workers=self.hparams.num_workers, 
    #         pin_memory=True
    #     )
    
class BRATSLatentsDataModule(pl.LightningDataModule):
    def __init__(self,
        autoencoder,
        latent_shape    = (1, 32, 32),
        to_2d           = True,
        root_path       = './data/brats_preprocessed.npy',
        npy_path        = './data/brats_preprocessed_latents.npy',
        n_samples       = 500,
        save_npy        = True,
        batch_size      = 32,
        shuffle         = True,
        num_workers     = 6,
        **kwargs
    ) -> None:
        assert (os.path.exists(root_path) or os.path.exists(npy_path)), 'Provide at least one valid data source!'
        super().__init__()
        self.autoencoder = autoencoder
        self.prepare_data_per_node = True # prepare_data executed only once on master node
        self.save_hyperparameters(ignore=['autoencoder'])       
        
    def prepare_data(self) -> None:
        if not os.path.exists(self.hparams.npy_path):
            print('Encoding dataset into latents ...')
            data = np.load(self.hparams.root_path)[:self.hparams.n_samples]
            
            # depth first as we can encode a whole 3D model at once
            data = data.transpose(0, 4, 1, 2, 3)
            
            # encoding the data into latents
            self.latents = np.zeros(shape=(
                self.hparams.n_samples,
                data.shape[1], # number of slices (Positions)
                *self.hparams.latent_shape
            ))
            
            self.autoencoder.eval()
            for idx in tqdm(range(self.hparams.n_samples), position=0, leave=True):
                input = torch.from_numpy(data[idx]).to(device, dtype=torch.float32)
                
                with torch.no_grad():
                    # encode a whole volume at once
                    pos = torch.arange(0, data.shape[1], device=device, dtype=torch.long)
                    pemb = self.autoencoder.encode_position(pos)
                    if isinstance(self.autoencoder, GaussianAutoencoder):
                        z = self.autoencoder.encode(input, pemb).sample()
                    elif isinstance(self.autoencoder, VQAutoencoder):
                        z, _ = self.autoencoder.encode_pre_quantization(input, pemb)
                    else:
                        raise NotImplementedError
                    
                    self.latents[idx] = z.cpu().numpy() # outputs will be of shape 4x32x32
                    
            # putting channels first 64x4x32x32 -> 4x64x32x32
            self.latents = self.latents.transpose(0, 2, 1, 3, 4)
            
            # => to 2D
            if self.hparams.to_2d:
                N, C, D, W, H = self.latents.shape
                w = h = int(np.sqrt(D * W * H))
                self.latents = self.latents.reshape(N, C, w, h)
                # grid_w = grid_h = int(np.sqrt(D))
                # self.latents = self.latents.reshape(B, C, grid_w, grid_h, W, H)
                # self.latents = self.latents.transpose(0, 1, 2, 4, 3, 5)
                # self.latents = self.latents.reshape(B, C, grid_w * W, grid_h * H)
                    
            print('Saving dataset as npy file...')
            np.save(self.hparams.npy_path, self.latents)
            print('Min latents:', self.latents.min())
            print('Max latents:', self.latents.max())
            print('Saved!')
            
        else:
            print('Dataset already exists at {}'.format(self.hparams.npy_path))
            
    def setup(self, stage='fit'):
        assert os.path.exists(self.hparams.npy_path), 'npy data file does not exist!'
        
        print('Loading dataset from npy file...')
        data = np.load(self.hparams.npy_path, allow_pickle=True)
        self.latents = data[:self.hparams.n_samples]
        print('Latents shape:', self.latents.shape)
        self.dataset = IdentityDataset(self.latents)
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.hparams.batch_size,
            shuffle=self.hparams.shuffle,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            drop_last=True
        )  