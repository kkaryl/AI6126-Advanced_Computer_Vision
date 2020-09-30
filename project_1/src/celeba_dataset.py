import os
import torch
import torch.utils.data as data
import PIL


class CelebaDataset(data.Dataset):
    def __init__(self, img_dir, ann_file, transform=None, target_transform=None):
        images = []
        targets = []

        for line in open(ann_file, 'r'):
            sample = line.split()
            if len(sample) != 41:
                raise (RuntimeError("# Annotated face attributes of CelebA dataset should not be different from 40"))
            images.append(sample[0])
            targets.append([int(i) for i in sample[1:]])

        self.images = [os.path.join(img_dir, img) for img in images]
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        path = self.images[index]
        sample = PIL.Image.open(path)
        target = self.targets[index]
        target = torch.LongTensor(target)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.images)