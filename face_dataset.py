import os
from PIL import ImageDraw
import shutil

import h5py
import numpy as np
import scipy.misc
import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms

from util import data_util
import json

def train_transform(opt):
    return transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize(opt.img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def test_transform(opt):
    return transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize(opt.img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

class Dataset(data.Dataset):
    def __init__(self, file_names, image_transform, max_size):
        self.transform = image_transform
        size = min(len(file_names), max_size)
        self.file_names = file_names[:size]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, item):
        image = Image.open(self.file_names[item])
        image = self.transform(image)
        return image


def process_landmark(landmark):
    x0,y0 = landmark[:,0],landmark[:,1]
    x1,y1 = x0,y0-20
    x2,y2 = (x1-88)/88,(y1-88)/88
    return torch.stack([x2,y2],dim=1)

class PairedCelebA(data.Dataset):
    def __init__(self, file_names,
                 image_transform, max_size,
                 attr_dict, sel_attr_names,
                 sampler, data_dir,landmark_dict):
        self.transform = image_transform
        size = min(len(file_names), max_size)
        self.file_names = file_names[:size]
        self.attr_dict = attr_dict
        self.set_of_files = [[]] * len(sel_attr_names)
        for idx, sel_attr in enumerate(sel_attr_names):
            for file_name in self.file_names:
                if self.attr_dict[file_name][idx] == 1:
                    self.set_of_files[idx].append(file_name)
            print("Attr {} with {} images".format(sel_attr,
                                                  len(self.set_of_files[idx])))
        self.sampler = sampler
        self.data_dir = data_dir
        self.sel_attr_names = sel_attr_names
        self.landmark_dict = landmark_dict

    def __getitem__(self, item):
        idx = 0
        while item >= len(self.set_of_files[idx]):
            item -= len(self.set_of_files[idx])
            idx += 1
        img_file = self.set_of_files[idx][item]
        sel_attr_name = self.sel_attr_names[idx]
        paired_file = self.sampler(item, sel_attr_name)
        img1 = Image.open(os.path.join(self.data_dir, img_file))
        img2 = Image.open(os.path.join(self.data_dir, paired_file))
        label1 = torch.FloatTensor(self.attr_dict[img_file])
        label2 = torch.FloatTensor(self.attr_dict[paired_file])
        landmark1 = torch.FloatTensor(self.landmark_dict[img_file])
        landmark2 = torch.FloatTensor(self.landmark_dict[paired_file])
        return self.transform(img2), label2,process_landmark(landmark2),\
               self.transform(img1), label1,process_landmark(landmark1)

    def __len__(self):
        total_num_files = 0
        for files in self.set_of_files:
            total_num_files += len(files)
        return total_num_files


class CelebAforTest(data.Dataset):
    def __init__(self, image_transform, max_size,
                 sel_attr_name,
                 attr_data_dir,
                 id_data_dir):
        self.transform = image_transform
        self.attr_dict = data_util.extract_attr_dict(data_util.ANNO_ROOT,
                                                     [sel_attr_name])
        attr_list = os.listdir(attr_data_dir)
        id_list = os.listdir(id_data_dir)
        size = min(max_size, len(attr_list), len(id_list))
        attr_list = attr_list[:size]
        id_list = id_list[:200]
        self.file_pair = []
        self.id_data_dir = id_data_dir
        self.attr_data_dir = attr_data_dir
        for id_img in id_list:
            for attr_img in attr_list:
                self.file_pair.append((id_img, attr_img))

    def __getitem__(self, item):
        id_fname, attr_fname = self.file_pair[item]
        id_img = Image.open(os.path.join(self.id_data_dir, id_fname))
        attr_img = Image.open(os.path.join(self.attr_data_dir, attr_fname))
        id_label = torch.FloatTensor([0])
        attr_label = torch.FloatTensor([1])
        return self.transform(id_img), id_label, \
               self.transform(attr_img), attr_label

    def __len__(self):
        return len(self.file_pair)


class MultiAttrSampler(object):
    def __init__(self, attr_dict, sel_attrs):
        self.sel_attrs = sel_attrs
        self.attr_dict = attr_dict
        self.attr_cache = {}
        for idx, attr_name in enumerate(self.sel_attrs):
            attr_list, non_attr_list = [], []
            for fname in self.attr_dict:
                if self.attr_dict[fname][idx] == 1:
                    attr_list.append(fname)
                else:
                    non_attr_list.append(fname)
            non_attr_list = non_attr_list[:len(attr_list)]
            self.attr_cache[attr_name] = (attr_list, non_attr_list)

    def __call__(self, idx, sel_attr_name):
        attr_list, non_attr_list = self.attr_cache[sel_attr_name]
        return non_attr_list[idx]


def parse_group(infilename):
    with open(infilename) as infile:
        lines = infile.readlines()
    group1 = []
    group2 = []
    for line in lines:
        name1, name2 = line.strip().split()
        group1.append(name1)
        group2.append(name2)
    return group1, group2


def create_attr_file_dir(sel_attr, data_root, max_size=1000):
    attr_dict = data_util.extract_attr_dict(data_util.ANNO_ROOT,
                                            [sel_attr])
    if not os.path.exists(data_root):
        os.makedirs(data_root)
        os.makedirs(os.path.join(data_root, 'attr_files'))
        os.makedirs(os.path.join(data_root, 'id_files'))
    attr_count,id_count = 0,0
    _, val_list = data_util.extract_train_val_fnames(data_util.EVAL_ROOT)
    np.random.shuffle(val_list)
    for fname in val_list:
        if attr_dict[fname][0] == 1:
            attr_count+=1
            if attr_count<max_size:
                shutil.copy2(os.path.join(data_util.DATA_ROOT, fname),
                         os.path.join(data_root, 'attr_files', fname))
        else:
            id_count+=1
            if id_count<max_size:
                shutil.copy2(os.path.join(data_util.DATA_ROOT, fname),
                         os.path.join(data_root, 'id_files', fname))
        print("{}/{}".format(min(attr_count,id_count), max_size))
        if min(attr_count,id_count) > max_size:
            break


def create_hd_attr_file_dir(sel_attr, data_root, max_size):
    from PIL import Image
    attr_dict = data_util.extract_attr_dict(data_util.ANNO_ROOT,
                                            [sel_attr])
    hd_mapping = data_util.extract_celeba_hq_mapping()
    h5file = h5py.File(data_util.CelebA_HQ_ROOT, 'r')
    img_dset = h5file['data1024x1024']

    if not os.path.exists(data_root):
        os.makedirs(data_root)
        os.makedirs(os.path.join(data_root, 'attr_files'))
        os.makedirs(os.path.join(data_root, 'id_files'))
    train_list,val_list = data_util.extract_train_val_fnames(data_util.EVAL_ROOT)
    val_list = [fname for fname in train_list if fname in hd_mapping]
    np.random.shuffle(val_list)
    attr_count,id_count = 0,0
    for fname in val_list:
        img_array = img_dset[hd_mapping[fname]]
        img_array = np.moveaxis(img_array,[0,1,2],[2,0,1])
        if attr_dict[fname][0] == 1:
            save_name = os.path.join(data_root, 'attr_files', fname)
            attr_count+=1
            if attr_count<max_size:
                Image.fromarray(img_array).save(save_name)
        else:
            save_name = os.path.join(data_root, 'id_files', fname)
            id_count+=1
            if id_count<max_size:
                Image.fromarray(img_array).save(save_name)
        print("{}/{}".format(min(attr_count,id_count), max_size))
        if min(attr_count,id_count) > max_size:
            break


def create_dataloader_celeba_test(file_root, max_size,
                                  img_size,
                                  sel_attr,
                                  is_hd,
                                  batch_size=1,
                                  num_workers=1):
    SEL_ATTRS = [sel_attr]
    assert len(SEL_ATTRS) == 1
    if not is_hd:
        transform = transforms.Compose([
            transforms.CenterCrop(178),
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    dataset = CelebAforTest(transform, max_size, sel_attr,
                            os.path.join(file_root, 'attr_files'),
                            os.path.join(file_root, 'id_files'))
    dataloader = data.DataLoader(dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers
                                 )
    return dataloader


def create_dataloader_celeba(opt):
    # SEL_ATTRS = ["Eyeglasses"]
    SEL_ATTRS = opt.sel_attrs
    train_list, val_list = data_util.extract_train_val_fnames(data_util.EVAL_ROOT)
    with open(data_util.LANDMARK_ROOT) as infile:
        landmark_dict = json.load(infile)
    train_list = [img for img in train_list if img in landmark_dict]
    val_list = [img for img in val_list if img in landmark_dict]
    if opt.train:
        transform = train_transform(opt)
        img_list = train_list
    else:
        transform = test_transform(opt)
        img_list = val_list

    attr_dict = data_util.extract_attr_dict(data_util.ANNO_ROOT, SEL_ATTRS)
    attr_dict = {img:attr_dict[img] for img in attr_dict if img in landmark_dict}
    sampler = MultiAttrSampler(attr_dict, SEL_ATTRS)
    dataset = PairedCelebA(img_list, transform,
                           10000000, attr_dict,
                           SEL_ATTRS, sampler,
                           data_util.DATA_ROOT,
                           landmark_dict)
    dataloader = data.DataLoader(dataset,
                                 batch_size=opt.batch_size,
                                 shuffle=True,
                                 num_workers=opt.nThreads
                                 )
    return dataloader
