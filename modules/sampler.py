import numpy as np
import torch

class ScheduleSampler:
    def __init__(self, 
        T,
        batch_size,
        sampler,
        **kwargs
    ) -> None:
        assert sampler in ['uniform', 'loss-aware'], "sampler must be either uniform or loss_aware"
        self.T = T
        self.batch_size = batch_size
        self.sampler = sampler

        # additional states
        if self.sampler == 'loss-aware':
            self.memory_span = kwargs.get('memory_span', 10)
            self.losses = np.zeros(shape=(T, self.memory_span), dtype=np.float64)

    def sample(self, device):
        """Importance sampling schedule"""
        p = self.weights()
        indices = np.random.choice(self.T, size=self.batch_size, p=p)
        weights = 1 / (self.T * p[indices])
        return (
            torch.from_numpy(indices).to(device, dtype=torch.long), 
            torch.from_numpy(weights).to(device, dtype=torch.float64)
        )
    
    def weights(self):
        """Compute weights for importance sampling"""
        if self.sampler == 'uniform':
            return np.ones(shape=(self.T,), dtype=np.float64) / self.T
        
        elif self.sampler == 'loss-aware':
            # warmup phase
            if (self.losses == 0.).any():
                return np.ones(shape=(self.T,), dtype=np.float64) / self.T

            # compute weights
            std = np.std(self.losses, axis=1)
            w = np.sqrt(np.mean(self.losses ** 2, axis=1)) + np.sqrt(std)
            return w / w.sum()
        
    def update_losses(self, ts, losses):
        """Update losses for loss-aware sampling"""
        for t, loss in zip(ts, losses):
            self.losses[t, :-1] = self.losses[t, 1:]
            self.losses[t, -1] = loss.item()


