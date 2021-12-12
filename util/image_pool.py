import random
import numpy as np
import torch
from torch.autograd import Variable
from collections import deque

class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.sample_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = deque()

    def add(self, images):
        if self.pool_size == 0:
            return images
        for image in images.data:
            image = torch.unsqueeze(image, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
            else:
                self.images.popleft()
                self.images.append(image)

    def query(self):
        if len(self.images) > self.sample_size:
            return_images = list(random.sample(self.images, self.sample_size))
        else:
            return_images = list(self.images)
        return torch.cat(return_images, 0)


# class ImagePool():
#     def __init__(self, pool_size):
#         self.pool_size = pool_size
#         if self.pool_size > 0:
#             self.num_imgs = 0
#             self.images = []
#
#     def query(self, images):
#         if self.pool_size == 0:
#             return images
#         return_images = []
#         for image in images.data:
#             image = torch.unsqueeze(image, 0)
#             if self.num_imgs < self.pool_size:
#                 self.num_imgs = self.num_imgs + 1
#                 self.images.append(image)
#                 return_images.append(image)
#             else:
#                 p = random.uniform(0, 1)
#                 if p > 0.5:
#                     random_id = random.randint(0, self.pool_size-1)
#                     tmp = self.images[random_id].clone()
#                     self.images[random_id] = image
#                     return_images.append(tmp)
#                 else:
#                     return_images.append(image)
#         return_images = Variable(torch.cat(return_images, 0))
#         return return_images
