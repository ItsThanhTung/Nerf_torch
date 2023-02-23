from src.dataloader.llff import get_data_loader
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)


if __name__ == "__main__":
    train_loader, test_loader = get_data_loader(1024, device)

    for (data, target) in train_loader:
        print(data)
