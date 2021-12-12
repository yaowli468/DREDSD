import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import numpy as np


class AlignedDataset(BaseDataset):
    def __init__(self, opt):
        # super(AlignedDataset,self).__init__(opt)
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)

        self.AB_paths = sorted(make_dataset(self.dir_AB))

        transform_list = [transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5),
                                                 (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        AB = self.transform(AB)

        w_total = AB.size(2)
        w = int(w_total / 2)
        h = AB.size(1)
        w_offset = random.randint(0, max(0, w - self.opt.fineWidthSize - 1))
        h_offset = random.randint(0, max(0, h - self.opt.fineheightSize - 1))

        w1_offset = random.randint(0, max(0, w - self.opt.fineWidthSize - 1))
        h1_offset = random.randint(0, max(0, h - self.opt.fineheightSize - 1))


        A = AB[:, h_offset:h_offset + self.opt.fineheightSize,   # Blurred
               w_offset:w_offset + self.opt.fineWidthSize]
        B = AB[:, h_offset:h_offset + self.opt.fineheightSize,    # Ground Truth
               w + w_offset:w + w_offset + self.opt.fineWidthSize]

        E_blur = AB[:, h1_offset:h1_offset + self.opt.fineheightSize,
             w1_offset:w1_offset + self.opt.fineWidthSize]   #Exemplar corresponding blurred

        E = AB[:, h1_offset:h1_offset + self.opt.fineheightSize,   #Exemplar
            w + w1_offset:w + w1_offset + self.opt.fineWidthSize]


        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)
            E = E.index_select(2, idx)
            E_blur = E_blur.index_select(2, idx)

        return {'A': A, 'B': B,'E': E,'E_blur':E_blur,
                'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'AlignedDataset'
