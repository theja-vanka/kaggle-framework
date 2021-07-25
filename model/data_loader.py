import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from helpers.config import Config


class CustomDataset(Dataset):
    def __init__(self, frame, transform=None):
        self.df = frame
        self.file_names = self.df['img_path'].values
        self.labels = self.df[Config.target_col].values
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def _extract_data(self, waves):
        pass

    def __getitem__(self, idx):
        file_path = self.file_names[idx]
        image = np.load(file_path).astype(np.float32)
        image = self._extract_data(image)
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.labels[idx]).float()
        return image, label


class Data_Loaders():

    def __init__(self, df, fold, batch_size):
        self.dataset = df
        self.train = df[df['kfold'] != fold].reset_index(drop=True)
        self.valid = df[df['kfold'] == fold].reset_index(drop=True)
        self.train_set = CustomDataset(self.train)
        self.valid_set = CustomDataset(self.valid)
        self.shuffle_dataset = True
        self.random_seed = Config.seed
        self.batch_size = batch_size

        self.train_loader = DataLoader(
                self.train_set,
                batch_size=batch_size,
                shuffle=self.shuffle_dataset,
                drop_last=True,
                pin_memory=True
            )
        self.test_loader = DataLoader(
                self.valid_set,
                batch_size=batch_size,
                shuffle=self.shuffle_dataset,
                drop_last=True,
                pin_memory=True
            )


def main():
    batch_size = 16
    data_loaders = Data_Loaders(batch_size)
    # note this is how the dataloaders will be iterated over to test
    for idx, (inputs, labels) in enumerate(data_loaders.train_loader):
        _, _ = inputs, labels
    for idx, (inputs, labels) in enumerate(data_loaders.test_loader):
        _, _ = inputs, labels


if __name__ == "__main__":
    main()