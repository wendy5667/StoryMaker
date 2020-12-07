# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 08:16:32 2018

@author: USER
"""

import torch
import torch.nn
import torch.optim as optim
import torchvision.transforms as Transform
from torchvision.utils import save_image
import numpy as np
import os
from tqdm import tqdm
# import demoWebsite.datasets
import demoWebsite.ACGAN as ACGAN
import demoWebsite.utils as utils
from torch.utils import data
# import pandas as pd
import sys

class Dataset(data.Dataset):
    def __init__(self, tags):
        self.tags = tags

    def __len__(self):
        return len(self.tags)

    def __getitem__(self, index):       
        return torch.tensor(self.tags[index]).float()

def create_img(tags):        
    hair_classes, eye_classes, face_classes, glasses_classes = 6, 4, 3, 2
    num_classes = hair_classes + eye_classes + face_classes + glasses_classes
    latent_dim = 100
    batch_size = 128

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    G_path = './demoWebsite/best.ckpt'

    G = ACGAN.Generator(latent_dim = latent_dim, class_dim = num_classes)
    prev_state = torch.load(G_path) if torch.cuda.is_available() else torch.load(G_path ,map_location='cpu')
    G.load_state_dict(prev_state['model'])
    G.eval()
    G.to(device)
    # testing_data = pd.read_csv('{}'.format(argv[1]), sep=" ", skiprows=1)
    # tags = testing_data.values.tolist()
#     with open('../data/tests.pickle', 'rb') as file:
#         tags = pkl.load(file)
    data_set = Dataset(tags)
    data_generator = data.DataLoader(data_set, batch_size=batch_size, shuffle=False, num_workers=10)
    
    all_img = []
    
    for batch_tag in tqdm(data_generator):
        batch_tag = batch_tag.to(device)
        z = torch.randn(batch_tag.size(0), latent_dim).to(device)
        img = G(z, batch_tag)
        img = img.detach().to('cpu') if torch.cuda.is_available() else img.detach()
        # all_img.append(img.detach().to('cpu'))
        all_img.append(img.detach())
        
    all_img = torch.cat(all_img, 0)
    img_path = []
    for i, _ in enumerate(all_img):
        save_image(utils.denorm(all_img[i]), './{}/{}.png'.format("templates/static", i))
        img_path.append('./{}/{}.png'.format("static",i))
    return img_path
    
    
if __name__ == "__main__":
    tags = [[0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0], [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0], [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0], [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1], [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1]]
    all_img = create_img(tags)
    