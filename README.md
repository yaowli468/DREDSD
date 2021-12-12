<!-- TITLE -->
### Deep Ranking Exemplar-based Dynamic Scene Deblurring 
By Yaowei Li, Jinshan Pan, Ye Luo, Jianwei Lu
 
## Dependencies
* Linux(Tested on Ubuntu 18.04) 
* Python 3.7 (Recomend to use [Anaconda](https://www.anaconda.com/products/individual#linux))
* Pytorch 1.8.0 (`conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge`)
* CUDA 11.1
* numpy (`conda install numpy`)
* PIL (`conda install pillow`)
* dominate (`conda install dominate`) 
* cv2 (`conda install opencv-python`)
* visdom (`conda install visdom`)

## Get Started

### Download
* Pretrained model can be downloaded from [HERE](https://pan.baidu.com/s/10097ea2xIdQ836b3VAQvjw)(7ci5), please put them to './checkpoints/experiment_name/'

## Testing
1. Run the following commands to test on our method.
 * If the exemplar has the corresponding blur:
    Firstly, combine the paired blur-sharp exemplar:
    ```sh
	cd ./pre_datasets
    python combine_blur_and_sharp.py --fold_A ./Blur --fold_B ./Sharp --fold_AB ./testSet/exemplar_AB
    ```
    Then, test image:
    ```sh
    python test.py --dataroot=./testSet/testImage --exemplar_dir=./testSet/exemplar_AB
    ```
 * If the exemplar has no coresponding blur:
    ```sh
    python test.py --no_blur_exemplar --dataroot=./testSet/testImage --exemplar_dir=./testSet/exemplar_AB
    ```
  After `test.py` is done, run `changeDeblurImage.py` to obtain the final resuts.
  
2. Run the following commands to test on our multi-scale method.
 * If the exemplar has the corresponding blur:
   Firstly, combine the paired blur-sharp exemplar:
   ```sh
   cd ./pre_datasets   
   python combine_blur_and_sharp.py --fold_A ./Blur --fold_B ./Sharp --fold_AB ./testSet/exemplar_AB
   ```
   Then, test image:
   ```sh
   python multi_test.py --dataroot=./testSet/testImage --exemplar_dir=./testSet/exemplar_AB
   ```
 * If the exemplar has no coresponding blur:
   ```sh
   python multi_test.py --no_blur_exemplar --dataroot=./testSet/testImage --exemplar_dir=./testSet/exemplar_AB
   ```
After `multi_test.py` is done, run `changeDeblurImage.py` to obtain the final resuts.

### Training
1. Run the following command to prepare dataset.
   ```sh
   cd ./pre_datasets  
   python combine_blur_and_sharp.py --fold_A ./Blur --fold_B ./Sharp --fold_AB ./Combined_Blur_Sharp
   ```
2. Run the following command to train the model.
   ```sh
   python train.py --dataroot ./pre_datasets/Combined_Blur_Sharp
   ```

## Acknowledgments
This code is based on [DeblurGAN](https://github.com/KupynOrest/DeblurGAN) and [CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). Thanks for their greate works.


