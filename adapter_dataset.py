import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os, json
import glob
import re
import math
from PIL import Image, ImageFile
import numpy as np



class FeatureDataset(Dataset):
    def __init__(self, data_json, num_object, label_offset=0):
        if os.path.exists(data_json):
            if data_json.endswith('.json'):
                with open(data_json, 'r') as f:
                    feat_dict = json.load(f)
                self.data = torch.Tensor(feat_dict['features'])#.cuda()
            elif data_json.endswith('.pth'):
                features = torch.load(data_json)
                emb_dim = features.size(-1)
                self.data = features.view(-1, emb_dim).float().cuda()
                print("Shape of descriptor tensor: ", self.data.size())
        self.num_template_per_object = self.data.size(0) // num_object
        print(f'num_template_per_object: {self.num_template_per_object}')
        self.label_offset = label_offset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_feature = self.data[index]
        label = index // self.num_template_per_object + self.label_offset  # 100 objects in total

        return img_feature, label

class ObjectFeatureDataset(Dataset):
    def __init__(self, data_json, num_object, label_offset=0):
        if os.path.exists(data_json):
            if data_json.endswith('.json'):
                with open(data_json, 'r') as f:
                    feat_dict = json.load(f)
                self.data = torch.Tensor(feat_dict['features'])#.cuda()
            elif data_json.endswith('.pth'):
                features = torch.load(data_json)
                emb_dim = features.size(-1)
                self.data = features.view(-1, emb_dim).float().cuda()
                print("Shape of descriptor tensor: ", self.data.size())
        self.num_template_per_object = self.data.size(0) // num_object
        print(f'num_template_per_object: {self.num_template_per_object}')
        self.label_offset = label_offset
        self.data = self.data.view(num_object, self.num_template_per_object, -1)
        # print(f'Shape of data: {self.data.size()}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_feature = self.data[index]
        cur_label = index + self.label_offset  # 100 objects in total
        label = [cur_label] * self.num_template_per_object
        label = torch.Tensor(label).long()

        return img_feature, label

class TokenDataset(Dataset):
    def __init__(self, data_dir, num_template_per_object, label_offset=0):
        self.data = sorted(glob.glob(os.path.join(data_dir, '*')))
        self.num_template_per_object = num_template_per_object
        self.label_offset = label_offset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        json_path = self.data[index]
        with open(json_path, 'r') as f:
            feat_dict = json.load(f)

        token = torch.Tensor(feat_dict['token']).cuda()  # torch tensor of shape (H, W, C)
        token = token.permute(2, 0, 1)  # torch tensor of shape (C, H, W)
        mask = torch.Tensor(feat_dict['mask']).cuda() # torch tensor of shape (1, H, W)
        label = index // self.num_template_per_object + self.label_offset  # 100 objects in total

        return token, mask, label

