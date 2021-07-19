# Template to train a model.
from helpers.config import Config
import random
import os
import numpy as np
import torch


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


if Config.debug:
    Config.epochs = 1
    # train = train.sample(n=50000, random_state=CFG.seed).reset_index(drop=True)

if __name__ == "__main__":
    seed_torch(seed=Config.seed)

