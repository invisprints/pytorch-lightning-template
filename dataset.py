import pytorch_lightning as pl
import torchvision.transforms as T
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets


class VisionDataModule(pl.LightningDataModule):

    train_transform = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomAffine(degrees=45, translate=(0.15, 0.15), scale=(0.9, 1.1)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def __init__(self, data_path="data",
                 batch_size=32,
                 **kwargs):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("vision data module")
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--data_path', type=str, default='./data')

        return parent_parser

    def prepare_data(self):
        # download and save
        assert os.path.exists(self.data_path), '{} does not exist'.format(self.data_path)

    def setup(self, stage):
        if stage == 'fit' or stage is None:
            self.train_dataset = datasets.MNIST(self.data_path, train=True, download=True,
                                               transform=self.train_transform)
            self.val_datasets = datasets.MNIST(self.data_path, train=False,
                                               transform=self.val_transform)
        elif stage == 'test' or stage is None:
            self.test_dataset = datasets.MNIST(self.data_path, train=False,
                                               transform=self.val_transform)


    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=4,
                          pin_memory=True,
                          drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=4)
    # def teardown(self, stage: Optional[str] = None):
        # Used to clean-up when the run is finished