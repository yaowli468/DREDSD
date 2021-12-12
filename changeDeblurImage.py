import os
from PIL import Image

if __name__ == '__main__':
    file_path='./results/experiment_name/test_latest/images'
    save_path='./results/deblur'
    for root,dirs,files in os.walk(file_path):
        for file in files:
            image_path=os.path.join(root, file)
            if str(file).split('.')[-2].split('_')[-2]=='fake':
                img=Image.open(image_path)
                img_name=str(file).split('.')[-2].split('_')[0]+str('.png')
                img_path=os.path.join(save_path,img_name)
                img.save(img_path)