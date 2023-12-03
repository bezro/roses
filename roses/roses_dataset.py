from pickle import load

import dvc.api
from torch.utils.data import Dataset


def load_roses_dataset(remote_dvc):
    # myremote
    with dvc.api.open("data/X.pkl", mode="rb", remote=remote_dvc) as file_X:
        X = load(file=file_X)
    with dvc.api.open("data/y.pkl", mode="rb", remote=remote_dvc) as file_y:
        y = load(file=file_y)
    return X, y


class RosesDataset(Dataset):
    def __init__(self, X, y, mask) -> None:
        super().__init__()

        self.X = X[mask].astype("float32")
        self.y = y[mask].astype("float32")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
