import numpy as np
from PIL import Image
import os
import io
import json
import glob
import h5py

from torchmeta.utils.data import Dataset, ClassDataset, CombinationMetaDataset
from torchvision.datasets.utils import download_url
from torchmeta.datasets.utils import get_asset
from torchmeta.transforms import ClassSplitter, Categorical, Rotation
from torchvision.transforms import ToTensor, Resize, Compose
from torchmeta.utils.data import BatchMetaDataLoader
from tqdm import tqdm
import pickle
import torch
import cv2
import sys

class VGGflower(CombinationMetaDataset):

    def __init__(self, root, num_classes_per_task=None, meta_train=False,
                 meta_val=False, meta_test=False, meta_split=None,
                 transform=None, target_transform=None, dataset_transform=None,
                 class_augmentations=None, download=False):
        dataset = VGGflowerClassDataset(root, meta_train=meta_train, meta_val=meta_val,
            meta_test=meta_test, meta_split=meta_split, transform=transform,
            class_augmentations=class_augmentations, download=download)
        super(VGGflower, self).__init__(dataset, num_classes_per_task,
            target_transform=target_transform, dataset_transform=dataset_transform)


class VGGflowerClassDataset(ClassDataset):
    folder = 'vgg_flower'
    filename_labels = '{0}_labels.json'
    def __init__(self, root, meta_train=False, meta_val=False, meta_test=False,
                 meta_split=None, transform=None, class_augmentations=None,
                 download=False):
        super(VGGflowerClassDataset, self).__init__(meta_train=meta_train,
            meta_val=meta_val, meta_test=meta_test, meta_split=meta_split,
            class_augmentations=class_augmentations)

        self.root = os.path.join(os.path.expanduser(root), self.folder)
        self.transform = transform
  
        self.split_filename_labels = os.path.join(self.root,
            self.filename_labels.format(self.meta_split))

        self._data_file = None
        self._data = None
        self._labels = None   
        self._num_classes = len(self.labels)

        #print('self.meta_split', self.meta_split)
    def __getitem__(self, index):
        label = self.labels[index % self.num_classes]
        transform = self.get_transform(index, self.transform)
        target_transform = self.get_target_transform(index)
        class_dict = torch.load(self.root + '/'+self.meta_split  + '/' + '{}.pt'.format(label))
        data = class_dict[label]
        
        return VGGflowerDataset(index, data, label, transform=transform,
                          target_transform=target_transform)

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def labels(self):
        if self._labels is None:
            with open(self.split_filename_labels, 'r') as f:
                self._labels = json.load(f)
        return self._labels


class VGGflowerDataset(Dataset):
    def __init__(self, index, data, label,
                 transform=None, target_transform=None):
        super(VGGflowerDataset, self).__init__(index, transform=transform,
                                         target_transform=target_transform)
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        image = self.data[index]
        target = self.label
        if self.target_transform is not None:
            target = self.target_transform(target)

        image = image.permute(1, 2, 0)
        image = image.data.numpy()
        #print('image', image.shape)
        #filename = '/media/cchen/StorageDisk/ZhenyiWang/Onlinemeta/Online_memory/dataloaderdir/temp/save1.jpg'
        #cv2.imwrite(filename, image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        #filename = '/media/cchen/StorageDisk/ZhenyiWang/Onlinemeta/Online_memory/dataloaderdir/temp/save2.jpg'
        #cv2.imwrite(filename, image)

        image = torch.tensor(image).unsqueeze(0)
        image = image.repeat(3, 1, 1)
        #print('image.shape', image.shape)
        #sys.exit(0)

        return (image, target)






def VGGflowerload(args):
    data_path = '/media/cchen/StorageDisk/ZhenyiWang/Meta_Task/data/'
    
    #Plantae
    dataset_transform = ClassSplitter(shuffle=True,
                                      num_train_per_class=args.num_shot,
                                      num_test_per_class=args.num_query)

    transform = None
    VGGflower_train_dataset = VGGflower(args.data_path,
                                      transform=transform,
                                      target_transform=Categorical(num_classes=args.num_way),
                                      num_classes_per_task=args.num_way,
                                      meta_train=True,
                                      dataset_transform=dataset_transform,
                                      download=False)


    VGGflower_train_loader = BatchMetaDataLoader(VGGflower_train_dataset, batch_size=args.VGGflower_batch_size,
        shuffle=True, pin_memory=True, num_workers=args.num_workers)

    VGGflower_val_dataset = VGGflower(args.data_path,
                                    transform=transform,
                                    target_transform=Categorical(num_classes=args.num_way),
                                    num_classes_per_task=args.num_way,
                                    meta_val=True,
                                    dataset_transform=dataset_transform)

    VGGflower_valid_loader = BatchMetaDataLoader(VGGflower_val_dataset, batch_size=args.valid_batch_size,
        shuffle=True, pin_memory=True, num_workers=args.num_workers)

    VGGflower_test_dataset = VGGflower(args.data_path,
                                     transform=transform,
                                     target_transform=Categorical(num_classes=args.num_way),
                                     num_classes_per_task=args.num_way,
                                     meta_test=True,
                                     dataset_transform=dataset_transform)
    VGGflower_test_loader = BatchMetaDataLoader(VGGflower_test_dataset, batch_size=args.valid_batch_size,
        shuffle=True, pin_memory=True, num_workers=args.num_workers)


    with tqdm(VGGflower_train_loader, total=300) as pbar:
            for batch_idx, batch in enumerate(pbar):
                train_inputs, train_targets = batch['train']
                if batch_idx>= 300:
                    break

    with tqdm(VGGflower_test_loader, total=20) as pbar:
            for batch_idx, batch in enumerate(pbar):
                train_inputs, train_targets = batch['train']
                if batch_idx>= 20:
                    break
                #print('batch_idx', batch_idx)
                #print('train_inputs.size()', train_inputs.size())





if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('Prototypical Networks')

    parser.add_argument('--data_path', type=str, default='/media/cchen/StorageDisk/ZhenyiWang/Meta_Task/data/',
        help='Path to the folder the data is downloaded to.')
    parser.add_argument('--num-shot', type=int, default=5,
        help='Number of examples per class (k in "k-shot", default: 5).')
    parser.add_argument('--num-way', type=int, default=5,
        help='Number of classes per task (N in "N-way", default: 5).')

    parser.add_argument('--embedding-size', type=int, default=64,
        help='Dimension of the embedding/latent space (default: 64).')
    parser.add_argument('--hidden-size', type=int, default=64,
        help='Number of channels for each convolutional layer (default: 64).')

    parser.add_argument('--output_folder', type=str, default='output/datasset/',
        help='Path to the output folder for saving the model (optional).')
    parser.add_argument('--batch-size', type=int, default=5,
        help='Number of tasks in a mini-batch of tasks (default: 16).')
    parser.add_argument('--MiniImagenet_batch_size', type=int, default=2,
        help='Number of tasks in a mini-batch of tasks for MiniImagenet (default: 4).')
    parser.add_argument('--CIFARFS_batch_size', type=int, default=2,
        help='Number of tasks in a mini-batch of tasks for CIFARFS (default: 4).')
    parser.add_argument('--CUB_batch_size', type=int, default=2,
        help='Number of tasks in a mini-batch of tasks for CUB (default: 4).')
    parser.add_argument('--Omniglot_batch_size', type=int, default=2,
        help='Number of tasks in a mini-batch of tasks for Omniglot (default: 4).')
    parser.add_argument('--Aircraft_batch_size', type=int, default=2,
        help='Number of tasks in a mini-batch of tasks for Aircraft (default: 4).')
    parser.add_argument('--Cars_batch_size', type=int, default=2,
        help='Number of tasks in a mini-batch of tasks for Aircraft (default: 4).')
    parser.add_argument('--Plantae_batch_size', type=int, default=2,
        help='Number of tasks in a mini-batch of tasks for Aircraft (default: 4).')
    parser.add_argument('--Quickdraw_batch_size', type=int, default=2,
        help='Number of tasks in a mini-batch of tasks for Quickdraw (default: 4).')
    parser.add_argument('--VGGflower_batch_size', type=int, default=2,
        help='Number of tasks in a mini-batch of tasks for VGGflower (default: 4).')
    parser.add_argument('--num_train_batches', type=int, default=200,
        help='Number of batches the prototypical network is trained over (default: 100).')
    parser.add_argument('--num_valid_batches', type=int, default=150,
        help='Number of batches the model is trained over (default: 150).')
    parser.add_argument('--num_memory_batches', type=int, default=1,
        help='Number of batches the model is trained over (default: 150).')
    parser.add_argument('--num-workers', type=int, default=1,
        help='Number of workers for data loading (default: 1).')
    parser.add_argument('--num_query', type=int, default=10,
        help='Number of query examples per class (k in "k-query", default: 15).')
    parser.add_argument('--download', action='store_true',
        help='Download the Omniglot dataset in the data folder.')
    parser.add_argument('--use-cuda', action='store_true',
        help='Use CUDA if available.')
    parser.add_argument('--ewc', action='store_true',
        help='Use EWC.')
    parser.add_argument('--ewcweight', type=float, default=1e4,
        help='Use EWC weight.')
    parser.add_argument('--num_epoch', type=int, default=200,
        help='Number of epochs for meta train.') 
    parser.add_argument('--valid_batch_size', type=int, default=5,
        help='Number of tasks in a mini-batch of tasks for validation (default: 4).')
    parser.add_argument('--grad-clipping', type=float, default=100.0)

    parser.add_argument('--clip_hyper', type=float, default=10.0)
    parser.add_argument('--LR', type=float, default=0.2)#learning rate regularization
    parser.add_argument('--hyper-lr', type=float, default=1e-4)
    parser.add_argument('--alpha', type=float, default=1e-6)
    parser.add_argument('--mu', type=float, default= 0.0)
    parser.add_argument('--first-order', action='store_true')
    parser.add_argument('--gpu', type=int, nargs='+', default=[1], help='0 = CPU.')

    args = parser.parse_args()
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print('args.device', args.device)
    VGGflowerload(args)

