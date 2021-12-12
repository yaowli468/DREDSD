import pdb

import torch
import torch.nn as nn
import functools
import numpy as np
from torchvision import models
from torch.nn import functional as F
from torch.optim import lr_scheduler

from .myresnet import MyResNet,BasicBlock

up_kwargs = {'mode': 'bilinear', 'align_corners': True}

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

def define_G(which_model_netG, gpu_ids=[]):
    if which_model_netG == 'RankAttentionDeblureNet':
        netG = RankAttentionDeblureNet()
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    if len(gpu_ids) > 0:
        netG.cuda(gpu_ids[0])
    netG.apply(weights_init)
    return netG

def define_D(input_nc, ndf, which_model_netD, n_layers_D=3, norm='batch', use_sigmoid=False, gpu_ids=[],
             use_parallel=True):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert (torch.cuda.is_available())
    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid,
                                   gpu_ids=gpu_ids, use_parallel=use_parallel)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid,
                                   gpu_ids=gpu_ids, use_parallel=use_parallel)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % which_model_netD)
    if use_gpu:
        netD.cuda(gpu_ids[0])
    netD.apply(weights_init)
    return netD

def define_trained_blur(gpu_ids=[]):

    TrainedBlur= TrainedBlurNetwork()
    if len(gpu_ids) > 0:
        TrainedBlur.cuda(gpu_ids[0])
    TrainedBlur.apply(weights_init)
    return TrainedBlur


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


#MyResnetUnet
def convrelu(in_channels, out_channels, kernel,stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, stride,padding=padding,bias=True),
        nn.ReLU(inplace=True),
    )

def _myresnet(arch, block, layers, **kwargs):
    model = MyResNet(block, layers, **kwargs)
    return model

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[],
                 use_parallel=True):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        self.use_parallel = use_parallel

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = int(np.ceil((kw - 1) / 2))
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor) and self.use_parallel:
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

class TrainedBlurNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        model=_myresnet('resnet18', BasicBlock, [1, 1, 1, 1])
        self.Elayer1_conv=model.conv1
        self.Elayer1_relu=model.relu

        self.Elayer1=model.layer1
        self.Elayer2=model.layer2
        self.Elayer3=model.layer3
        self.Elayer4=model.layer4

        self.Dlayer4=model.layer5
        self.Dlayer3=model.layer6
        self.Dlayer2 = model.layer7
        self.Dlayer1 = model.layer8

        self.upsample3=nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upsample2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upsample1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.upsample0 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_last = nn.Conv2d(32, 3, 1)

        self.tanh = nn.Tanh()

    def forward(self,input_current):

        input_Elayer1 = self.Elayer1_conv(input_current)
        input_Elayer1 = self.Elayer1_relu(input_Elayer1)
        input_Elayer1 = self.Elayer1(input_Elayer1)
        input_Elayer2 = self.Elayer2(input_Elayer1)
        input_Elayer3 = self.Elayer3(input_Elayer2)
        feature_current = self.Elayer4(input_Elayer3)

        out_Dlayer4 = self.Dlayer4(feature_current)

        out_Dlayer3 = self.upsample3(out_Dlayer4)
        out_Dlayer3=input_Elayer3+out_Dlayer3

        out_Dlayer3 = self.Dlayer3(out_Dlayer3)
        out_Dlayer2 = self.upsample2(out_Dlayer3)

        out_Dlayer2=input_Elayer2+out_Dlayer2

        out_Dlayer2 = self.Dlayer2(out_Dlayer2)
        out_Dlayer1 = self.upsample1(out_Dlayer2)
        out_Dlayer1=input_Elayer1+out_Dlayer1

        out_Dlayer1 = self.Dlayer1(out_Dlayer1)
        out = self.upsample0(out_Dlayer1)
        out = self.conv_last(out)

        return self.tanh(out)


class RankAttentionDeblureNet(nn.Module):
    def __init__(self):
        super().__init__()

        model=_myresnet('resnet18', BasicBlock, [3, 3, 3, 3])
        self.Elayer1_conv=model.conv1
        self.Elayer1_relu=model.relu

        self.Elayer1=model.layer1
        self.Elayer2=model.layer2
        self.Elayer3=model.layer3
        self.Elayer4=model.layer4

        self.Dlayer4=model.layer5
        self.Dlayer3=model.layer6
        self.Dlayer2 = model.layer7
        self.Dlayer1 = model.layer8

        self.upsample3=nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.upsample2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.upsample1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.upsample0 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_last = nn.Conv2d(32, 3, 1)

        self.tanh = nn.Tanh()

        self.conv_con_channel4=nn.Conv2d(512+512,512,kernel_size=3,stride=1,padding=1,bias=True)
        self.conv_con_channel3=nn.Conv2d(256+256+256,256,kernel_size=3,stride=1,padding=1,bias=True)
        self.conv_con_channel2=nn.Conv2d(128+128+128,128,kernel_size=3,stride=1,padding=1,bias=True)
        self.conv_con_channel1=nn.Conv2d(64+64,64,kernel_size=3,stride=1,padding=1,bias=True)

        self.conv_f_512=convrelu(256,512,5,2,2)
        self.conv_f_256 = convrelu(128,256,5,2,2)
        self.conv_f_128 = convrelu(64,128,5,2,2)
        self.conv_f_64 = convrelu(3,64,5,2,2)


    def gather(self, input, dim, index):
        views = [input.size(0)] + [1 if i != dim else -1 for i in range(1, len(input.size()))]
        expanse = list(input.size())
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(input, dim, index)

    def RankingCorrelationModule(self,f_current,f_refer,f_refer_blur):
        feature_current_unfold = F.unfold(f_current, kernel_size=(3, 3), padding=1)
        feature_refer_unfold = F.unfold(f_refer, kernel_size=(3, 3), padding=1)
        feature_refer_blur_unfold = F.unfold(f_refer_blur, kernel_size=(3, 3), padding=1)
        feature_refer_blur_unfold = feature_refer_blur_unfold.permute(0, 2, 1)

        feature_refer_blur_unfold = F.normalize(feature_refer_blur_unfold, dim=2)
        feature_current_unfold = F.normalize(feature_current_unfold, dim=1)

        relation_refer_current = torch.bmm(feature_refer_blur_unfold, feature_current_unfold)
        max_value, max_index = torch.max(relation_refer_current, dim=1)

        feature_refer_max_unfold = self.gather(feature_refer_unfold, dim=2, index=max_index)

        feature_current_fold = F.fold(feature_refer_max_unfold, output_size=f_current.size()[-2:],
                                      kernel_size=(3, 3), padding=1)

        feature_cat = feature_current_fold+f_current
        return feature_cat

    def forward(self,input_current,real_current,real_current_blur):
        input_Elayer1 = self.Elayer1_conv(input_current)
        input_Elayer1 = self.Elayer1_relu(input_Elayer1)
        input_Elayer1 = self.Elayer1(input_Elayer1)
        input_Elayer2 = self.Elayer2(input_Elayer1)
        input_Elayer3 = self.Elayer3(input_Elayer2)
        feature_current = self.Elayer4(input_Elayer3)

        real_Elayer1_blur = self.Elayer1_conv(real_current_blur)
        real_Elayer1_blur = self.Elayer1_relu(real_Elayer1_blur)
        real_Elayer1_blur = self.Elayer1(real_Elayer1_blur)
        real_Elayer2_blur = self.Elayer2(real_Elayer1_blur)
        real_Elayer3_blur = self.Elayer3(real_Elayer2_blur)
        feature_refer_blur = self.Elayer4(real_Elayer3_blur)

        real_Elayer1=self.conv_f_64(real_current)
        real_Elayer2=self.conv_f_128(real_Elayer1)
        real_Elayer3=self.conv_f_256(real_Elayer2)
        feature_refer=self.conv_f_512(real_Elayer3)

        f_rank_correlation4=self.RankingCorrelationModule(feature_current,feature_refer,feature_refer_blur)
        f_rank_correlation3=self.RankingCorrelationModule(input_Elayer3,real_Elayer3,real_Elayer3_blur)
        f_rank_correlation2 = self.RankingCorrelationModule(input_Elayer2, real_Elayer2, real_Elayer2_blur)

        out_Dlayer4=self.conv_con_channel4(torch.cat((f_rank_correlation4,feature_current),dim=1))
        out_Dlayer4 = self.Dlayer4(out_Dlayer4)

        out_Dlayer3 = self.upsample3(out_Dlayer4)
        out_Dlayer3=self.conv_con_channel3(torch.cat((input_Elayer3,out_Dlayer3,f_rank_correlation3),dim=1))
        out_Dlayer3 = self.Dlayer3(out_Dlayer3)

        out_Dlayer2 = self.upsample2(out_Dlayer3)
        out_Dlayer2= self.conv_con_channel2(torch.cat((input_Elayer2,out_Dlayer2,f_rank_correlation2),dim=1))
        out_Dlayer2 = self.Dlayer2(out_Dlayer2)

        out_Dlayer1 = self.upsample1(out_Dlayer2)
        out_Dlayer1=self.conv_con_channel1(torch.cat((input_Elayer1,out_Dlayer1),dim=1))
        out_Dlayer1 = self.Dlayer1(out_Dlayer1)

        out = self.upsample0(out_Dlayer1)
        out = self.conv_last(out)

        return self.tanh(out)