import os
import numpy as np
import cv2
import argparse

# python combine_blur_and_sharp.py --fold_A ./Blur --fold_B ./Sharp --fold_AB ./Combined_Blur_Sharp

parser = argparse.ArgumentParser('create image pairs')
parser.add_argument('--fold_A', dest='fold_A', help='input directory for image A', type=str, default='./Blur')
parser.add_argument('--fold_B', dest='fold_B', help='input directory for image B', type=str, default='./Sharp')
parser.add_argument('--fold_AB', dest='fold_AB', help='output directory', type=str, default='./Combined_Blur_Sharp')
parser.add_argument('--num_imgs', dest='num_imgs', help='number of images',type=int, default=1000000)
args = parser.parse_args()

for arg in vars(args):
    print('[%s] = ' % arg,  getattr(args, arg))

img_fold_B=args.fold_B
img_fold_A=args.fold_A
img_fold_AB=args.fold_AB
img_list=os.listdir(img_fold_A)
num_imgs=min(args.num_imgs, len(img_list))
if not os.path.isdir(img_fold_AB):
    os.makedirs(img_fold_AB)
numberImage=1
for n in range(num_imgs):
    print("........."+str(numberImage))
    name_A = img_list[n]
    path_A = os.path.join(img_fold_A, name_A)
    name_B = name_A
    path_B = os.path.join(img_fold_B, name_B)
    if os.path.isfile(path_A) and os.path.isfile(path_B):
        name_AB = name_A
        path_AB = os.path.join(img_fold_AB, name_AB)
        im_A = cv2.imread(path_A, cv2.IMREAD_COLOR)
        im_B = cv2.imread(path_B, cv2.IMREAD_COLOR)
        im_AB = np.concatenate([im_A, im_B], 1)
        cv2.imwrite(path_AB, im_AB)
        numberImage=numberImage+1
