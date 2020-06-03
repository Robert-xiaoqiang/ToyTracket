import torch.utils.data as data
from os.path import join
import cv2
import json
import numpy as np


class VID(data.Dataset):
    def __init__(self, file, root, range=10, train=True):
        self.imdb = json.load(open(file, 'r'))
        self.root = root
        self.range = range
        self.train = train
        self.mean = np.expand_dims(np.expand_dims(np.array([109, 120, 119]), axis=1), axis=1).astype(np.float32)

    def __getitem__(self, item):
        if self.train:
            target_id = self.imdb['train_set'][item]
        else:
            target_id = self.imdb['val_set'][item]

        range_up = self.imdb['up_index'][target_id]
        search_id1, search_id2 = np.random.choice(np.arange(1, min(range_up, self.range+1)), 2, replace = False) + target_id
        target = cv2.imread(join(self.root, '{:08d}.jpg'.format(target_id)))
        search1 = cv2.imread(join(self.root, '{:08d}.jpg'.format(search_id1)))
        search2 = cv2.imread(join(self.root, '{:08d}.jpg'.format(search_id2)))

        target = np.transpose(target, (2, 0, 1)).astype(np.float32) - self.mean
        search1 = np.transpose(search1, (2, 0, 1)).astype(np.float32) - self.mean
        search2 = np.transpose(search2, (2, 0, 1)).astype(np.float32) - self.mean
        print(target.shape, search1.shape, search2.shape)
        return target, search1, search2

    def __len__(self):
        if self.train:
            return len(self.imdb['train_set'])
        else:
            return len(self.imdb['val_set'])


class MGTVTrainVID(data.Dataset):
    def __init__(self, file, root, range=10):
        self.imdb = json.load(open(file, 'r'))
        self.root = root
        self.range = range
        self.mean = np.expand_dims(np.expand_dims(np.array([109, 120, 119]), axis=1), axis=1).astype(np.float32)

    def __getitem__(self, item):
        target_id = self.imdb['train_set'][item]

        range_up = self.imdb['up_index'][target_id]
        upper_bound = min(range_up, self.range+1)
        subsequent_idx = np.arange(1, upper_bound)
        search_id1, *left_over = np.random.choice(subsequent_idx, 2 if upper_bound > 4 else 1, replace = False) + target_id
        search_id2 = left_over[0] if len(left_over) else search_id1
        target = cv2.imread(join(self.root, '{:05d}.jpg'.format(target_id)))
        search1 = cv2.imread(join(self.root, '{:05d}.jpg'.format(search_id1)))
        search2 = cv2.imread(join(self.root, '{:05d}.jpg'.format(search_id2)))

        target = np.transpose(target, (2, 0, 1)).astype(np.float32) - self.mean
        search1 = np.transpose(search1, (2, 0, 1)).astype(np.float32) - self.mean
        search2 = np.transpose(search2, (2, 0, 1)).astype(np.float32) - self.mean
        
        return target, search1, search2

    def __len__(self):
        return len(self.imdb['train_set'])


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    data = VID(train=True)
    n = len(data)
    fig = plt.figure(1)
    ax = fig.add_axes([0, 0, 1, 1])

    for i in range(n):
        z, x = data[i]
        z, x = np.transpose(z, (1, 2, 0)).astype(np.uint8), np.transpose(x, (1, 2, 0)).astype(np.uint8)
        zx = np.concatenate((z, x), axis=1)

        ax.imshow(cv2.cvtColor(zx, cv2.COLOR_BGR2RGB))
        p = patches.Rectangle(
            (125/3, 125/3), 125/3, 125/3, fill=False, clip_on=False, linewidth=2, edgecolor='g')
        ax.add_patch(p)
        p = patches.Rectangle(
            (125 / 3+125, 125 / 3), 125 / 3, 125 / 3, fill=False, clip_on=False, linewidth=2, edgecolor='r')
        ax.add_patch(p)
        plt.pause(0.5)
