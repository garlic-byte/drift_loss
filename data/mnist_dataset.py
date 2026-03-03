from collections import defaultdict

import torch
from torchvision import datasets, transforms
from torch.utils.data import IterableDataset
import random


class MnistDataset(IterableDataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        transform = transforms.Compose([
            transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # [-1, 1]
        ])
        self.train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        self.num_classes = len(self.train_dataset.class_to_idx)
        self.index = 0
        self.min_index = 0
        self.vessels = defaultdict(list)

    def __len__(self):
        return len(self.train_dataset) // self.num_classes


    def prepare_data(self):
        """Classification of MNIST dataset."""
        # step 1. shutil datas from datasets.
        dataset_labels = self.train_dataset.targets
        schedule = [(data_index, int(data_label)) for data_index, data_label in enumerate(dataset_labels)]
        random.shuffle(schedule)

        # step 2. split datas into different vessels according labels.
        vessels = defaultdict(list)
        for data_index, data_label in schedule:
            vessels[data_label].append(data_index)

        min_index = float('inf')
        for data_label in vessels:
            min_index = min(min_index, len(vessels[data_label]))

        self.vessels = vessels
        self.min_index = min_index
        self.index = 0

    def get_step_data(self, indices):
        step_data = []
        step_labels = []
        for index in indices:
            data, label = self.train_dataset[index]
            step_data.append(data)
            step_labels.append(label)
        step_data = torch.stack(step_data)
        step_labels = torch.tensor(step_labels)
        return step_data, step_labels


    def __iter__(self):
        """
        Return batch of data and corresponding labels,
        data [num_classes, img_channels, img_height, img_width],
        labels [num_classes]
        """
        self.prepare_data()
        while True:
            if self.index >= self.min_index:
                self.prepare_data()
            batch_data = []
            batch_label = []
            for label in range(self.num_classes):
                data_index = self.vessels[label][self.index]
                step_data, step_labels = self.train_dataset[data_index]
                batch_data.append(step_data)
                batch_label.append(step_labels)
            yield torch.stack(batch_data, dim=0), torch.tensor(batch_label, dtype=torch.long)
            self.index += 1


if __name__ == '__main__':
    dataset = MnistDataset("/home/wsj/Desktop/data/dataset_origin/mnist")
    for d1, lab in dataset:
        assert d1.shape == (320, 1, 28, 28)
        assert lab.shape == (320,)
