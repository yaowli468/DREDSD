import os
import cv2
import data_utils


def calc_image_PSNR_SSIM(ouput_root, gt_root, crop_border=4, test_ycbcr=False):
    '''
    计算图片的 PSNR、SSIM，使用 EDVR 的计算方式
    要求 ouput_root, gt_root 中的文件按顺序一一对应
    '''

    if test_ycbcr:
        print('Testing Y channel.')
    else:
        print('Testing RGB channels.')

    PSNR_list = []
    SSIM_list = []
    output_img_list = sorted(os.listdir(ouput_root))
    gt_img_list = sorted(os.listdir(gt_root))
    for o_im, g_im in zip(output_img_list, gt_img_list):
        o_im_path = os.path.join(ouput_root, o_im)
        g_im_path = os.path.join(gt_root, g_im)
        im_GT = cv2.imread(g_im_path) / 255.
        im_Gen = cv2.imread(o_im_path) / 255.

        h, w, c = im_Gen.shape
        im_GT = im_GT[:h, :w, :]  # crop GT to output size

        if test_ycbcr and im_GT.shape[2] == 3:  # evaluate on Y channel in YCbCr color space
            im_GT = data_utils.bgr2ycbcr(im_GT)
            im_Gen = data_utils.bgr2ycbcr(im_Gen)

        # crop borders
        if im_GT.ndim == 3:
            cropped_GT = im_GT[crop_border:-crop_border, crop_border:-crop_border, :]
            cropped_Gen = im_Gen[crop_border:-crop_border, crop_border:-crop_border, :]
        elif im_GT.ndim == 2:
            cropped_GT = im_GT[crop_border:-crop_border, crop_border:-crop_border]
            cropped_Gen = im_Gen[crop_border:-crop_border, crop_border:-crop_border]
        else:
            raise ValueError('Wrong image dimension: {}. Should be 2 or 3.'.format(im_GT.ndim))

        psnr = data_utils.PSNR_EDVR(cropped_GT * 255, cropped_Gen * 255)
        ssim = data_utils.SSIM_EDVR(cropped_GT * 255, cropped_Gen * 255)
        PSNR_list.append(psnr)
        SSIM_list.append(ssim)

        print("{} PSNR={:.5}, SSIM={:.4}".format(o_im, psnr, ssim))

    print('Average PSNR={:.5}, SSIM={:.4}'.format(sum(PSNR_list) / len(PSNR_list), sum(SSIM_list) / len(SSIM_list)))

    return PSNR_list, SSIM_list


def batch_calc_image_PSNR_SSIM(root_list, crop_border=4, test_ycbcr=False):
    for i, root in enumerate(root_list):
        ouput_root = root['output']
        gt_root = root['gt']
        print(">>>>  now test >>>>")
        print(">>>>  output: {}".format(ouput_root))
        print(">>>>  gt: {}".format(gt_root))
        calc_image_PSNR_SSIM(ouput_root, gt_root, crop_border=crop_border, test_ycbcr=test_ycbcr)

if __name__ == '__main__':
    ouput_root='../results/deblur'
    gt_root='../results/GT'
    calc_image_PSNR_SSIM(ouput_root=ouput_root ,gt_root=gt_root )
