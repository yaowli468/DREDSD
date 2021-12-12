import os.path
import pdb

import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random

class SingleDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        #self.root = opt.dataroot
        self.input = opt.dataroot
        self.exemplar=opt.exemplar_dir
        self.no_blur_exemplar=opt.no_blur_exemplar

        self.input_paths = make_dataset(self.input)
        self.exemplar_paths=make_dataset(self.exemplar)

        self.input_paths = sorted(self.input_paths)

        self.transform = get_transform(opt)

    def __getitem__(self, index):
        input_path = self.input_paths[index]
        input_img = Image.open(input_path).convert('RGB')
        input = self.transform(input_img)

        exemplar_path=random.choice(self.exemplar_paths)
        exemplar_img = Image.open(exemplar_path).convert('RGB')

        if self.no_blur_exemplar==True:
            exemplar = self.transform(exemplar_img)

            return {'input':input, 'exemplar': exemplar, 'input_path': input_path}

        else:
            exemplar_AB = self.transform(exemplar_img)
            w_total = exemplar_AB.size(2)
            w = int(w_total / 2)
            exemplar_A= exemplar_AB[:, :, 0:w]
            exemplar_B = exemplar_AB[:, :, w:w + w_total]

            return {'input':input, 'exemplar_A': exemplar_A, 'exemplar_B': exemplar_B, 'input_path': input_path}

    def __len__(self):
        return len(self.input_paths)

    def name(self):
        return 'SingleImageDataset'
