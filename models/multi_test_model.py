from torch.autograd import Variable
from collections import OrderedDict
import util.util as util
from .base_model import BaseModel
from . import multi_networks
import torch


class TestModel(BaseModel):
    def name(self):
        return 'TestModel'

    def __init__(self, opt):
        assert(not opt.isTrain)
        super(TestModel, self).__init__(opt)
        self.no_blur_exemplar=opt.no_blur_exemplar

        self.netG = multi_networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, self.gpu_ids, False,
                                      opt.learn_residual)

        self.trained_blur = multi_networks.define_trained_blur(self.gpu_ids)

        which_epoch = opt.which_epoch
        self.load_network(self.netG, 'MultiG', which_epoch)

        self.load_network(self.trained_blur, 'B', which_epoch)

        print('---------- Networks initialized -------------')
        multi_networks.print_network(self.netG)
        print('-----------------------------------------------')

    def set_input(self, input):
        self.input_blur = input['input'].cuda()
        self.image_paths = input['input_path']

        if self.no_blur_exemplar==True:
            self.exemplar = input['exemplar'].cuda()
        else:
            self.exemplar_blur = input['exemplar_A'].cuda()
            self.exemplar = input['exemplar_B'].cuda()

    def test(self):
        with torch.no_grad():
            if self.no_blur_exemplar==True:
                self.exemplar_blur = self.trained_blur.forward(self.exemplar)
                assert self.input_blur.size(-1)==self.exemplar.size(-1) and self.input_blur.size(-2)==self.exemplar.size(-2), 'input and exemplar should be same dimention'
                self.fake_B = self.netG.forward(self.input_blur, self.exemplar, self.exemplar_blur)

            else:
                assert self.input_blur.size(-1)==self.exemplar.size(-1) and self.input_blur.size(-2)==self.exemplar.size(-2), 'input and exemplar should be same dimention'
                self.fake_B = self.netG.forward(self.input_blur, self.exemplar, self.exemplar_blur)

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def get_current_visuals(self):
        real_A = util.tensor2im(self.input_blur.data)
        fake_B = util.tensor2im(self.fake_B.data)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B)])
