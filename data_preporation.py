from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image
import pandas as pd

import glob
from datetime import datetime


def collect_paths(train_path="idao_dataset/train/"):
    """ Create dict which will be converted to
    pandas DataFrame
    """
    image_paths = {
        item: 1. for item
        in glob.glob(train_path+"NR/*", recursive=False)
    }

    image_paths.update({
        item: 0. for item
        in glob.glob(train_path+"ER/*", recursive=False)
    })
    
    return image_paths


def build_df(source_dict):
    train_images = pd.DataFrame(
        data=source_dict.items(),
        columns=["path", "class"]
    )

    return train_images


class IDAODataset(Dataset):
    def __init__(self, table, transform=None):
        self.table = table
        self.transform = (
            transforms.ToTensor()
        )

    def __getitem__(self, index):
        path, label = self.table.iloc[index, :].to_list()
        image = Image.open(path).convert('L') # as grayscale
        image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.table)

    
def build_dataset(train_path):
    paths = collect_paths(train_path)
    df = build_df(paths)
    
    return IDAODataset(df)