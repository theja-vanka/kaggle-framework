# Template to train a model.
from helpers.config import Config
from helpers.utils import seed_torch

if Config.debug:
    Config.epochs = 1
    # train = train.sample(n=50000, random_state=CFG.seed).reset_index(drop=True)

if __name__ == "__main__":
    random_state = seed_torch(seed=Config.seed)
