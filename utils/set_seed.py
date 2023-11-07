import random

import numpy as np
import torch


class SetSeed:
    def __init__(self, seed):
        self.seed = seed

    def __enter__(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        return self

    def __exit__(self, *args):
        pass
