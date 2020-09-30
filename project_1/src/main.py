import os
import random
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
import models
import config
from celeba_dataset import CelebaDataset

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_loader = DataLoader(
    CelebaDataset(
        config.IMG_DIR,
        config.TRAIN_ATTRIBUTE_LIST,
        transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            normalize,
        ])),
    batch_size=config.train_batch, shuffle=True,
    num_workers=config.dl_workers, pin_memory=True)

val_loader = DataLoader(
    CelebaDataset(
        config.IMG_DIR,
        config.VAL_ATTRIBUTE_LIST,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
    batch_size=config.test_batch, shuffle=False,
    num_workers=config.dl_workers, pin_memory=True)

test_loader = DataLoader(
    CelebaDataset(
        config.IMG_DIR,
        config.TEST_ATTRIBUTE_LIST,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
    batch_size=config.test_batch, shuffle=False,
    num_workers=config.dl_workers, pin_memory=True)

# Initalize environment
if config.manual_seed is None:
    seed = random.randint(1, 10000)
random.seed(config.manual_seed)
torch.manual_seed(config.manual_seed)
use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.cuda.manual_seed_all(config.manual_seed)
device = torch.device("cuda" if use_cuda else "cpu")

if __name__ == '__main__':
    model_names = sorted(name for name in models.__dict__
                         if callable(models.__dict__[name])) # and name.islower() and not name.startswith("__"))
    print(model_names)
    # model = torchvision.models.resnet18(pretrained=True)
    # print(model)
