import os
import numpy as np
import torch.utils.data as data
import pandas as pd
from torchvision import transforms
from PIL import Image
from MixMatch.dataset.transforms import TransformTwice
from MixMatch.utils.misc import get_mean_and_std

class RafDataset(data.Dataset):
    def __init__(self, raf_path, dataset, labeled=True, transform=None):
        self.dataset = dataset
        self.labeled = labeled
        self.transform = transform
        self.raf_path = raf_path

    def get_image(self, img_name):
        img_path = os.path.join(self.raf_path, 'rafdb/aligned', img_name)
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __getitem__(self, idx):

        img_name = self.dataset.iloc[idx, 0]
        if not self.labeled:
            target = -1
        else:
            target = self.dataset.iloc[idx, 1]
        img = self.get_image(img_name)

        return img, target

    def __len__(self):
        return len(self.dataset)


def get_rafdb(root, n_labeled, training_type='ssl', transform_train=None, transform_val=None):

    base_dataset = pd.read_csv(os.path.join(root, 'rafdb/list_patition_label.txt'), sep=' ', header=None,
                               names=['img', 'label'])
    # change the names to actual names of the images and make labels start from 0
    add_align = lambda x: str(x).split('.')[0] + '_aligned.jpg'
    base_dataset['img'] = base_dataset['img'].apply(add_align)
    base_dataset['label'] = base_dataset['label'] - 1
    # Split the database in train, val, test
    dataset_train = base_dataset[base_dataset['img'].str.startswith('train')]
    targets_train = dataset_train.iloc[:, 1].values
    if training_type == 'ssl':
        train_labeled_idxs, train_unlabeled_idxs = train_split(targets_train, int(n_labeled/7))
        train_labeled = dataset_train.iloc[train_labeled_idxs, :]
        train_unlabeled = dataset_train.iloc[train_unlabeled_idxs, :]
        test = base_dataset[base_dataset['img'].str.startswith('test')]

        train_labeled_dataset = RafDataset(root, train_labeled, labeled=True, transform=transform_train)
        train_unlabeled_dataset = RafDataset(root, train_unlabeled, labeled=False, transform=TransformTwice(transform_train))
        test_dataset = RafDataset(root, test, labeled=True, transform=transform_val)
        return train_labeled_dataset, train_unlabeled_dataset, test_dataset
    else:
        dataset_train = dataset_train.sample(frac=1, random_state=42).reset_index(drop=True)
        train_labeled = dataset_train.iloc[:, :]
        test = base_dataset[base_dataset['img'].str.startswith('test')]

        train_labeled_dataset = RafDataset(root, train_labeled, labeled=True, transform=transform_train)
        test_dataset = RafDataset(root, test, labeled=True, transform=transform_val)
        return train_labeled_dataset, test_dataset


def train_split(labels, n_labeled_per_class):
    labels = np.array(labels)
    nr_classes = max(np.unique(labels)) + 1
    train_labeled_idxs = []
    train_unlabeled_idxs = []
    for i in range(nr_classes):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        train_labeled_idxs.extend(idxs[:n_labeled_per_class])
        train_unlabeled_idxs.extend(idxs[n_labeled_per_class:])
    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)

    return train_labeled_idxs, train_unlabeled_idxs


if __name__ == '__main__':
    root = '/home/demon/Alexandru/MixMatch_emotion/MixMatch/data/RafDB'
    train_labeled, train_unlabeled, val, test = get_rafdb(root=root, n_labeled=250)

    base_dataset = pd.read_csv(os.path.join(root, 'rafdb/list_patition_label.txt'), sep=' ', header=None,
                                   names=['img', 'label'])
    add_align = lambda x: str(x).split('.')[0] + '_aligned.jpg'
    base_dataset['img'] = base_dataset['img'].apply(add_align)
    transform = transforms.ToTensor()
    raf_dataset = RafDataset(raf_path=root, dataset=base_dataset, transform=transform)
    mean, std = get_mean_and_std(raf_dataset)
    print(f"mean:{mean}\n std:{std}")

