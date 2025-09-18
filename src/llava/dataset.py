import os
import pickle
from torch.utils.data import Dataset

class ImageMetDataset(Dataset):
    """
    Loads a pickle dataset of (image, caption) pairs.
    Expected format inside pickle: Dictionary with keys "train" and "val", where each value is in turn a list of dicts with keys {"image": <path>, "caption": <str>}
    """
    def __init__(self, data_path, split="train", transform=None):
        with open(data_path, "rb") as f:
            all_data = pickle.load(f)

        assert split in all_data, f"Split {split} not found in dataset. Available: {list(all_data.keys())}"
        self.data = all_data[split]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image, caption = item["image"], item["caption"]

        if self.transform:
            image = self.transform(image)

        return {"image": image, "caption": caption}
