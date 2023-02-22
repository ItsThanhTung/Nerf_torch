from src.dataloader.llff import LlffProcessor, LlffDataset
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)

data_processor = LlffProcessor()
train_data = data_processor.get_train_data()
test_data = data_processor.get_test_data()

LlffTrainData = LlffDataset(train_data, device)

print(LlffTrainData.__getitem__(1))