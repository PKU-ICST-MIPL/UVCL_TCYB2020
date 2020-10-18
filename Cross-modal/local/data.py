# -----------------------------------------------------------
# Stacked Cross Attention Network implementation based on 
# https://arxiv.org/abs/1803.08024.
# "Stacked Cross Attention for Image-Text Matching"
# Kuang-Huei Lee, Xi Chen, Gang Hua, Houdong Hu, Xiaodong He
#
# Writen by Kuang-Huei Lee, 2018
# ---------------------------------------------------------------
"""Data provider"""

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import nltk
from PIL import Image
import numpy as np
import json as jsonmod
import random


class PrecompDataset(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(self, data_path, data_split, vocab):
        self.vocab = vocab
        loc = data_path + '/'
        
        caps_path='/home/chijingze/sunhongbo/OpenNMT-py-master/flickr_fake_caps_random_60000.txt'
        IOU_caps_path='/media/zhuoyunkan/unsupervised_IOU/flickr_top_IOU_caps_862.txt'

        self.captions = []
        self.IOU_captions = []
        if data_split!='train':
            caps_path='/media/zhuoyunkan/unsupervised2019/data/f30k_precomp/%s_caps.txt' % data_split
            IOU_caps_path=caps_path
        with open(caps_path, 'rb') as f:
            for line in f:
                self.captions.append(line.strip())
        with open(IOU_caps_path, 'rb') as f:
            for line in f:
                self.IOU_captions.append(line.strip())
        
        self.img_entities_idx=np.load('/media/zhuoyunkan/unsupervised2019/data/flickr_image_entities_36/%s_img_entities_idx.npy' % data_split)
        self.caps_entities_idx=np.load('/media/zhuoyunkan/unsupervised2019/data/flickr_caps_entities/%s_caps_entities_idx.npy' % data_split)

        # Image features
        self.images = np.load(loc+'%s_ims.npy' % data_split)
                
        self.length = len(self.captions)
        
        print(self.images.shape)
        print(self.length)
        self.data_split = data_split
        if self.images.shape[0] != self.length:
            self.im_div = 5
        else:
            self.im_div = 1

    def __getitem__(self, index):
        IOU_ratio=0.5
        img_id = index/self.im_div
        image = torch.Tensor(self.images[img_id])
        if self.data_split=='test':
            image_entity_idx = torch.Tensor(self.img_entities_idx[img_id/5])
        else:
            image_entity_idx = torch.Tensor(self.img_entities_idx[img_id])
            
        caps_entity_idx = torch.Tensor(self.caps_entities_idx[index])
        
        caption = self.captions[index]
        if random.random()<IOU_ratio:
            caption = self.IOU_captions[index]
        
        vocab = self.vocab

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(
            str(caption).lower().decode('utf-8'))
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target, index, image_entity_idx, caps_entity_idx, img_id

    def __len__(self):
        return self.length


def collate_fn(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list of (image, caption) tuple.
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, ids, image_entity_idx, caps_entity_idx, img_ids = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)
    image_entity_idxs = torch.stack(image_entity_idx, 0)
    caps_entity_idxs = torch.stack(caps_entity_idx, 0)

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    return images, targets, lengths, image_entity_idxs, caps_entity_idxs, ids


def get_precomp_loader(data_path, data_split, vocab, opt, batch_size=100,
                       shuffle=True, num_workers=2):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    dset = PrecompDataset(data_path, data_split, vocab)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              collate_fn=collate_fn)
    return data_loader


def get_loaders(data_name, vocab, batch_size, workers, opt):
    dpath = os.path.join(opt.data_path, data_name)
    train_loader = get_precomp_loader(dpath, 'train', vocab, opt, 
                                      batch_size, True, workers)
    val_loader = get_precomp_loader(dpath, 'test', vocab, opt,
                                    batch_size, True, workers)
    return train_loader, val_loader 


def get_test_loader(split_name, data_name, vocab, batch_size,
                    workers, opt):
    dpath = os.path.join(opt.data_path, data_name)
    test_loader = get_precomp_loader(dpath, split_name, vocab, opt,
                                     batch_size, True, workers)
    return test_loader
