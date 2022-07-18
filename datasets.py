import numpy as np
import torch
from torchvision import transforms
from collections import OrderedDict
import os
from torchmeta.datasets import Omniglot, MiniImagenet, CIFARFS, CUB
from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.transforms import Categorical, Rotation
from torchvision.transforms import ToTensor, Resize, Compose
from dataloaderdir.CUB import *
from dataloaderdir.AIRCRAFTloader import *
from dataloaderdir.Plantae import *
from dataloaderdir.Quickdraw import *
from dataloaderdir.Vggflower import *
from dataloaderdir.Butterfly import *
from dataloaderdir.splitters import *



def dataset(args, datanames):
    #MiniImagenet   
    dataset_transform = ClassSplitter(shuffle=True,
                                      num_train_per_class=args.num_shot,
                                      num_test_per_class=args.num_query, 
                                      num_distractor_classes = args.num_distractor_classes)
    transform = Compose([Resize(84), ToTensor()])


    MiniImagenet_transform = ClassSplitter(shuffle=True,
                                      num_train_per_class=args.num_shot,
                                      num_test_per_class=args.num_query, 
                                      ratio = 0.4,
                                      num_distractor_classes = args.num_distractor_classes)

    MiniImagenet_train_dataset = MiniImagenet(args.data_path,
                                      transform=transform,
                                      target_transform=Categorical(num_classes=args.num_way),
                                      num_classes_per_task=args.num_way,
                                      meta_train=True,
                                      dataset_transform=MiniImagenet_transform,
                                      download=False)

    Imagenet_train_loader = BatchMetaDataLoader(MiniImagenet_train_dataset, batch_size=args.MiniImagenet_batch_size,
        shuffle=True, pin_memory=True, num_workers=args.num_workers)

    MiniImagenet_val_dataset = MiniImagenet(args.data_path,
                                    transform=transform,
                                    target_transform=Categorical(num_classes=args.num_way),
                                    num_classes_per_task=args.num_way,
                                    meta_val=True,
                                    dataset_transform=MiniImagenet_transform)

    Imagenet_valid_loader = BatchMetaDataLoader(MiniImagenet_val_dataset, batch_size=args.valid_batch_size,
        shuffle=True, pin_memory=True, num_workers=args.num_workers)

    MiniImagenet_test_dataset = MiniImagenet(args.data_path,
                                     transform=transform,
                                     target_transform=Categorical(num_classes=args.num_way),
                                     num_classes_per_task=args.num_way,
                                     meta_test=True,
                                     dataset_transform=MiniImagenet_transform)

    Imagenet_test_loader = BatchMetaDataLoader(MiniImagenet_test_dataset, batch_size=args.valid_batch_size,
        shuffle=True, pin_memory=True, num_workers=args.num_workers)


    #CIFARFS

    CIFARFS_transform = ClassSplitter(shuffle=True,
                                      num_train_per_class=args.num_shot,
                                      num_test_per_class=args.num_query, 
                                      ratio = 0.4,
                                      num_distractor_classes = args.num_distractor_classes)

    transform = Compose([Resize(84), ToTensor()])
    CIFARFS_train_dataset = CIFARFS(args.data_path,
                                      transform=transform,
                                      target_transform=Categorical(num_classes=args.num_way),
                                      num_classes_per_task=args.num_way,
                                      meta_train=True,
                                      dataset_transform=CIFARFS_transform,
                                      download=False)

    CIFARFS_train_loader = BatchMetaDataLoader(CIFARFS_train_dataset, batch_size=args.CIFARFS_batch_size,
        shuffle=True, pin_memory=True, num_workers=args.num_workers)

    CIFARFS_val_dataset = CIFARFS(args.data_path,
                                    transform=transform,
                                    target_transform=Categorical(num_classes=args.num_way),
                                    num_classes_per_task=args.num_way,
                                    meta_val=True,
                                    dataset_transform=CIFARFS_transform)

    CIFARFS_valid_loader = BatchMetaDataLoader(CIFARFS_val_dataset, batch_size=args.valid_batch_size,
        shuffle=True, pin_memory=True, num_workers=args.num_workers)

    CIFARFS_test_dataset = CIFARFS(args.data_path,
                                     transform=transform,
                                     target_transform=Categorical(num_classes=args.num_way),
                                     num_classes_per_task=args.num_way,
                                     meta_test=True,
                                     dataset_transform=CIFARFS_transform)
    CIFARFS_test_loader = BatchMetaDataLoader(CIFARFS_test_dataset, batch_size=args.valid_batch_size,
        shuffle=True, pin_memory=True, num_workers=args.num_workers)


    #CUB dataset
    
    CUB_transform = ClassSplitter(shuffle=True,
                                      num_train_per_class=args.num_shot,
                                      num_test_per_class=args.num_query, 
                                      ratio = 0.4,
                                      num_distractor_classes = args.num_distractor_classes)

    #transform = Compose([ToTensor()])
    transform = None
    #transform = Compose([Resize(84), ToTensor()])
    CUB_train_dataset = CUBdata(args.data_path,
                                      transform=transform,
                                      target_transform=Categorical(num_classes=args.num_way),
                                      num_classes_per_task=args.num_way,
                                      meta_train=True,
                                      dataset_transform=CUB_transform,
                                      download=False)

    CUB_train_loader = BatchMetaDataLoader(CUB_train_dataset, batch_size=args.CUB_batch_size,
        shuffle=True, pin_memory=True, num_workers=args.num_workers)

    CUB_val_dataset = CUBdata(args.data_path,
                                    transform=transform,
                                    target_transform=Categorical(num_classes=args.num_way),
                                    num_classes_per_task=args.num_way,
                                    meta_val=True,
                                    dataset_transform=CUB_transform)

    CUB_valid_loader = BatchMetaDataLoader(CUB_val_dataset, batch_size=args.valid_batch_size,
        shuffle=True, pin_memory=True, num_workers=args.num_workers)

    CUB_test_dataset = CUBdata(args.data_path,
                                     transform=transform,
                                     target_transform=Categorical(num_classes=args.num_way),
                                     num_classes_per_task=args.num_way,
                                     meta_test=True,
                                     dataset_transform=CUB_transform)
    CUB_test_loader = BatchMetaDataLoader(CUB_test_dataset, batch_size=args.valid_batch_size,
        shuffle=True, pin_memory=True, num_workers=args.num_workers)
    
    #Omniglot
    class_augmentations = [Rotation([90, 180, 270])]
    transform = Compose([Resize(84), ToTensor()])


 

    
    transform = None
    Aircraft_transform = ClassSplitter(shuffle=True,
                                      num_train_per_class=args.num_shot,
                                      num_test_per_class=args.num_query,
                                      ratio = 0.4,
                                      num_distractor_classes = args.num_distractor_classes)

    Aircraft_train_dataset = Aircraftdata(args.data_path,
                                      transform=transform,
                                      target_transform=Categorical(num_classes=args.num_way),
                                      num_classes_per_task=args.num_way,
                                      meta_train=True,
                                      dataset_transform=Aircraft_transform,
                                      download=False)


    Aircraft_train_loader = BatchMetaDataLoader(Aircraft_train_dataset, batch_size=args.Aircraft_batch_size,
        shuffle=True, pin_memory=True, num_workers=args.num_workers)

    Aircraft_val_dataset = Aircraftdata(args.data_path,
                                    transform=transform,
                                    target_transform=Categorical(num_classes=args.num_way),
                                    num_classes_per_task=args.num_way,
                                    meta_val=True,
                                    dataset_transform=Aircraft_transform)

    Aircraft_valid_loader = BatchMetaDataLoader(Aircraft_val_dataset, batch_size=args.valid_batch_size,
        shuffle=True, pin_memory=True, num_workers=args.num_workers)

    Aircraft_test_dataset = Aircraftdata(args.data_path,
                                     transform=transform,
                                     target_transform=Categorical(num_classes=args.num_way),
                                     num_classes_per_task=args.num_way,
                                     meta_test=True,
                                     dataset_transform=Aircraft_transform)
    Aircraft_test_loader = BatchMetaDataLoader(Aircraft_test_dataset, batch_size=args.valid_batch_size,
        shuffle=True, pin_memory=True, num_workers=args.num_workers)
    



    #Plantae
    transform = None
    Plantae_transform = ClassSplitter(shuffle=True,
                                      num_train_per_class=args.num_shot,
                                      num_test_per_class=args.num_query,
                                      ratio = 0.4,
                                      num_distractor_classes = args.num_distractor_classes)

    Plantae_train_dataset = Plantaedata(args.data_path,
                                      transform=transform,
                                      target_transform=Categorical(num_classes=args.num_way),
                                      num_classes_per_task=args.num_way,
                                      meta_train=True,
                                      dataset_transform=Plantae_transform,
                                      download=False)


    Plantae_train_loader = BatchMetaDataLoader(Plantae_train_dataset, batch_size=args.Plantae_batch_size,
        shuffle=True, pin_memory=True, num_workers=args.num_workers)

    Plantae_val_dataset = Plantaedata(args.data_path,
                                    transform=transform,
                                    target_transform=Categorical(num_classes=args.num_way),
                                    num_classes_per_task=args.num_way,
                                    meta_val=True,
                                    dataset_transform=Plantae_transform)

    Plantae_valid_loader = BatchMetaDataLoader(Plantae_val_dataset, batch_size=args.valid_batch_size,
        shuffle=True, pin_memory=True, num_workers=args.num_workers)

    Plantae_test_dataset = Plantaedata(args.data_path,
                                     transform=transform,
                                     target_transform=Categorical(num_classes=args.num_way),
                                     num_classes_per_task=args.num_way,
                                     meta_test=True,
                                     dataset_transform=Plantae_transform)
    Plantae_test_loader = BatchMetaDataLoader(Plantae_test_dataset, batch_size=args.valid_batch_size,
        shuffle=True, pin_memory=True, num_workers=args.num_workers)




    transform = None
    Butterfly_transform = ClassSplitter(shuffle=True,
                                      num_train_per_class=args.num_shot,
                                      num_test_per_class=args.num_query,
                                      ratio = 0.4,
                                      num_distractor_classes = args.num_distractor_classes)

    Butterfly_train_dataset = Butterfly(args.data_path,
                                      transform=transform,
                                      target_transform=Categorical(num_classes=args.num_way),
                                      num_classes_per_task=args.num_way,
                                      meta_train=True,
                                      dataset_transform=Butterfly_transform,
                                      download=False)


    Butterfly_train_loader = BatchMetaDataLoader(Butterfly_train_dataset, batch_size=args.Butterfly_batch_size,
        shuffle=True, pin_memory=True, num_workers=args.num_workers)

    Butterfly_val_dataset = Butterfly(args.data_path,
                                    transform=transform,
                                    target_transform=Categorical(num_classes=args.num_way),
                                    num_classes_per_task=args.num_way,
                                    meta_val=True,
                                    dataset_transform=Butterfly_transform)

    Butterfly_valid_loader = BatchMetaDataLoader(Butterfly_val_dataset, batch_size=args.valid_batch_size,
        shuffle=True, pin_memory=True, num_workers=args.num_workers)

    Butterfly_test_dataset = Butterfly(args.data_path,
                                     transform=transform,
                                     target_transform=Categorical(num_classes=args.num_way),
                                     num_classes_per_task=args.num_way,
                                     meta_test=True,
                                     dataset_transform=Butterfly_transform)
    Butterfly_test_loader = BatchMetaDataLoader(Butterfly_test_dataset, batch_size=args.valid_batch_size,
        shuffle=True, pin_memory=True, num_workers=args.num_workers)


    Omniglot_transform = ClassSplitter(shuffle=True,
                                      num_train_per_class=args.num_OODshot1,
                                      num_test_per_class=1,
                                      ratio = 0.9,
                                      num_distractor_classes = args.num_distractor_classes)

    transform = Compose([Resize(84), ToTensor()])
    Omniglot_train_dataset = Omniglot(args.data_path,
                                      transform=transform,
                                      target_transform=Categorical(num_classes=args.OODnum_way),
                                      num_classes_per_task=args.OODnum_way,
                                      meta_train=True,
                                      class_augmentations=class_augmentations,
                                      dataset_transform=Omniglot_transform,
                                      download=True)

    Omniglot_train_loader = BatchMetaDataLoader(Omniglot_train_dataset, batch_size=args.Omniglot_batch_size,
        shuffle=True, pin_memory=True, num_workers=args.num_workers)

    Omniglot_val_dataset = Omniglot(args.data_path,
                                    transform=transform,
                                    target_transform=Categorical(num_classes=args.num_way),
                                    num_classes_per_task=args.num_way,
                                    meta_val=True,
                                    class_augmentations=class_augmentations,
                                    dataset_transform=Omniglot_transform)

    Omniglot_valid_loader = BatchMetaDataLoader(Omniglot_val_dataset, batch_size=args.valid_batch_size,
        shuffle=True, pin_memory=True, num_workers=args.num_workers)

    Omniglot_test_dataset = Omniglot(args.data_path,
                                     transform=transform,
                                     target_transform=Categorical(num_classes=args.num_way),
                                     num_classes_per_task=args.num_way,
                                     meta_test=True,
                                     dataset_transform=Omniglot_transform)
    Omniglot_test_loader = BatchMetaDataLoader(Omniglot_test_dataset, batch_size=args.valid_batch_size,
        shuffle=True, pin_memory=True, num_workers=args.num_workers)




    transform = None
    Quickdraw_train_dataset = Quickdraw(args.data_path,
                                      transform=transform,
                                      target_transform=Categorical(num_classes=args.num_way),
                                      num_classes_per_task=args.num_way,
                                      meta_train=True,
                                      dataset_transform=dataset_transform,
                                      download=False)


    Quickdraw_train_loader = BatchMetaDataLoader(Quickdraw_train_dataset, batch_size=args.batch_size,
        shuffle=True, pin_memory=True, num_workers=args.num_workers)

    Quickdraw_val_dataset = Quickdraw(args.data_path,
                                    transform=transform,
                                    target_transform=Categorical(num_classes=args.num_way),
                                    num_classes_per_task=args.num_way,
                                    meta_val=True,
                                    dataset_transform=dataset_transform)

    Quickdraw_valid_loader = BatchMetaDataLoader(Quickdraw_val_dataset, batch_size=args.valid_batch_size,
        shuffle=True, pin_memory=True, num_workers=args.num_workers)

    Quickdraw_test_dataset = Quickdraw(args.data_path,
                                     transform=transform,
                                     target_transform=Categorical(num_classes=args.num_way),
                                     num_classes_per_task=args.num_way,
                                     meta_test=True,
                                     dataset_transform=dataset_transform)
    Quickdraw_test_loader = BatchMetaDataLoader(Quickdraw_test_dataset, batch_size=args.valid_batch_size,
        shuffle=True, pin_memory=True, num_workers=args.num_workers)

        

    VGGflower_transform = ClassSplitter(shuffle=True,
                                      num_train_per_class=args.num_OODshot2,
                                      num_test_per_class=args.num_query,
                                      ratio = 0.4,
                                      num_distractor_classes = args.num_distractor_classes)

    transform = None
    VGGflower_train_dataset = VGGflower(args.data_path,
                                      transform=transform,
                                      target_transform=Categorical(num_classes=args.OODnum_way),
                                      num_classes_per_task=args.OODnum_way,
                                      meta_train=True,
                                      dataset_transform=VGGflower_transform,
                                      download=False)


    VGGflower_train_loader = BatchMetaDataLoader(VGGflower_train_dataset, batch_size=args.VGGflower_batch_size,
        shuffle=True, pin_memory=True, num_workers=args.num_workers)

    VGGflower_val_dataset = VGGflower(args.data_path,
                                    transform=transform,
                                    target_transform=Categorical(num_classes=args.num_way),
                                    num_classes_per_task=args.num_way,
                                    meta_val=True,
                                    dataset_transform=VGGflower_transform)

    VGGflower_valid_loader = BatchMetaDataLoader(VGGflower_val_dataset, batch_size=args.valid_batch_size,
        shuffle=True, pin_memory=True, num_workers=args.num_workers)

    VGGflower_test_dataset = VGGflower(args.data_path,
                                     transform=transform,
                                     target_transform=Categorical(num_classes=args.num_way),
                                     num_classes_per_task=args.num_way,
                                     meta_test=True,
                                     dataset_transform=VGGflower_transform)
    VGGflower_test_loader = BatchMetaDataLoader(VGGflower_test_dataset, batch_size=args.valid_batch_size,
        shuffle=True, pin_memory=True, num_workers=args.num_workers)


    train_loader_list = []
    valid_loader_list = []
    test_loader_list = []
    for name in  datanames:
        if name == 'MiniImagenet':
            train_loader_list.append({name: Imagenet_train_loader})
            valid_loader_list.append({name: Imagenet_valid_loader})
            test_loader_list.append({name:  Imagenet_test_loader})
        if name == 'CIFARFS':
            train_loader_list.append({name:CIFARFS_train_loader})
            valid_loader_list.append({name:CIFARFS_valid_loader})
            test_loader_list.append({name:CIFARFS_test_loader})
        if name == 'CUB':
            train_loader_list.append({name:CUB_train_loader})
            valid_loader_list.append({name:CUB_valid_loader})
            test_loader_list.append({name:CUB_test_loader})
        if name == 'Aircraft':
            train_loader_list.append({name:Aircraft_train_loader})
            valid_loader_list.append({name:Aircraft_valid_loader})
            test_loader_list.append({name:Aircraft_test_loader})
        if name == 'Omniglot':
            train_loader_list.append({name:Omniglot_train_loader})
            valid_loader_list.append({name:Omniglot_valid_loader})
            test_loader_list.append({name:Omniglot_test_loader})
        
        if name == 'Plantae':
            train_loader_list.append({name:Plantae_train_loader})
            valid_loader_list.append({name:Plantae_valid_loader})
            test_loader_list.append({name:Plantae_test_loader})
        
        if name == 'Quickdraw':
            train_loader_list.append({name:Quickdraw_train_loader})
            valid_loader_list.append({name:Quickdraw_valid_loader})
            test_loader_list.append({name:Quickdraw_test_loader})

        if name == 'Vggflower':
            train_loader_list.append({name:VGGflower_train_loader})
            valid_loader_list.append({name:VGGflower_valid_loader})
            test_loader_list.append({name:VGGflower_test_loader})

        if name == 'Butterfly':
            train_loader_list.append({name:Butterfly_train_loader})
            valid_loader_list.append({name:Butterfly_valid_loader})
            test_loader_list.append({name:Butterfly_test_loader})

    return  train_loader_list, valid_loader_list, test_loader_list 


