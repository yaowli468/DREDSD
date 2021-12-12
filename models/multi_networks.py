import torch
import torch.nn as nn
import functools
import math
import numpy as np
from torchvision import models
from torch.nn import functional as F

from .myresnet import MyResNet,BasicBlock

up_kwargs = {'mode': 'bilinear', 'align_corners': True}

###############################################################################
# Functions
###############################################################################

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0.0, 0.5 * math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
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


def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, gpu_ids=[], use_parallel=True,
             learn_residual=False):
    netG = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert (torch.cuda.is_available())

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

def _myresnet(arch, block, layers, **kwargs):
    model = MyResNet(block, layers, **kwargs)
    return model


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
        model_v1 = _myresnet('resnet18', BasicBlock, [3, 3, 3, 3])
        model_v2 = _myresnet('resnet18', BasicBlock, [3, 3, 3, 3])
        model_v3 = _myresnet('resnet18', BasicBlock, [3, 3, 3, 3])

        self.Elayer_v1_conv = model_v1.conv1
        self.Elayer_v1_relu = model_v1.relu

        self.Elayer_v1_1 = model_v1.layer1
        self.Elayer_v1_2 = model_v1.layer2
        self.Elayer_v1_3 = model_v1.layer3
        #self.Elayer_v1_4 = model_v1.layer4

        #self.Dlayer_v1_4 = model_v1.layer5
        self.Dlayer_v1_3 = model_v1.layer6
        self.Dlayer_v1_2 = model_v1.layer7
        self.Dlayer_v1_1 = model_v1.layer8

        #self.upsample_v1_4 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upsample_v1_3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upsample_v1_2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upsample_v1_1 = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.conv_trans_chnannel_v1_3=nn.Conv2d(256 + 256+256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_trans_chnannel_v1_2=nn.Conv2d(128 + 128, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_trans_chnannel_v1_1=nn.Conv2d(64 + 64, 64, kernel_size=3, stride=1, padding=1, bias=True)


        self.Elayer_v2_conv = model_v2.conv1
        self.Elayer_v2_relu = model_v2.relu

        self.Elayer_v2_1 = model_v2.layer1
        self.Elayer_v2_2 = model_v2.layer2
        self.Elayer_v2_3 = model_v2.layer3
        #self.Elayer_v2_4 = model_v2.layer4

        #self.Dlayer_v2_4 = model_v2.layer5
        self.Dlayer_v2_3 = model_v2.layer6
        self.Dlayer_v2_2 = model_v2.layer7
        self.Dlayer_v2_1 = model_v2.layer8

        #self.upsample_v2_4 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upsample_v2_3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upsample_v2_2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upsample_v2_1 = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.conv_trans_chnannel_v2_3 = nn.Conv2d(256 + 256+256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_trans_chnannel_v2_2 = nn.Conv2d(128 + 128, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_trans_chnannel_v2_1 = nn.Conv2d(64 + 64, 64, kernel_size=3, stride=1, padding=1, bias=True)

        self.Elayer_v3_conv = model_v3.conv1
        self.Elayer_v3_relu = model_v3.relu

        self.Elayer_v3_1 = model_v3.layer1
        self.Elayer_v3_2 = model_v3.layer2
        self.Elayer_v3_3 = model_v3.layer3
        #self.Elayer_v3_4 = model_v3.layer4

        #self.Dlayer_v3_4 = model_v3.layer5
        self.Dlayer_v3_3 = model_v3.layer6
        self.Dlayer_v3_2 = model_v3.layer7
        self.Dlayer_v3_1 = model_v3.layer8

        #self.upsample_v3_4 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upsample_v3_3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upsample_v3_2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upsample_v3_1 = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.conv_trans_chnannel_v3_3 = nn.Conv2d(256 + 256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_trans_chnannel_v3_2 = nn.Conv2d(128 + 128, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_trans_chnannel_v3_1 = nn.Conv2d(64 + 64, 64, kernel_size=3, stride=1, padding=1, bias=True)

        self.tanh=nn.Tanh()

        #self.conv_trans_channel = nn.Conv2d(256 + 256, 256, kernel_size=3, stride=1, padding=1, bias=True)


    def gather(self, input, dim, index):
        views = [input.size(0)] + [1 if i != dim else -1 for i in range(1, len(input.size()))]
        expanse = list(input.size())
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(input, dim, index)


    def RankingCorrelationModule(self,feature_current,feature_refer,feature_refer_blur):
        feature_current_unfold = F.unfold(feature_current, kernel_size=(3, 3), padding=1)
        feature_refer_unfold = F.unfold(feature_refer, kernel_size=(3, 3), padding=1)
        feature_refer_blur_unfold = F.unfold(feature_refer_blur, kernel_size=(3, 3), padding=1)
        feature_refer_blur_unfold = feature_refer_blur_unfold.permute(0, 2, 1)

        feature_refer_blur_unfold = F.normalize(feature_refer_blur_unfold, dim=2)
        feature_current_unfold = F.normalize(feature_current_unfold, dim=1)

        relation_refer_current = torch.bmm(feature_refer_blur_unfold, feature_current_unfold)
        max_value, max_index = torch.max(relation_refer_current, dim=1)

        feature_refer_max_unfold = self.gather(feature_refer_unfold, dim=2, index=max_index)

        feature_current_fold = F.fold(feature_refer_max_unfold, output_size=feature_current.size()[-2:],
                                      kernel_size=(3, 3), padding=1)

        #feature_cat = torch.cat((feature_current_fold, feature_current), dim=1)
        #feature_cat = self.conv_trans_channel(feature_cat)
        feature_cat=feature_current+feature_current_fold

        return feature_cat


    def forward(self, input_current, real_current,real_current_blur):
        H=input_current.size(2)
        W=input_current.size(3)

        input_current_v3_LU=input_current[:,:,0:int(H/2),0:int(W/2)]
        input_current_v3_RU = input_current[:, :, 0:int(H / 2), int(W/2):W]
        input_current_v3_LD = input_current[:, :, int(H/2):H, 0:int(W/2)]
        input_current_v3_RD = input_current[:, :, int(H / 2):H, int(W/2):W]
        real_current_v3_LU = real_current[:, :, 0:int(H / 2), 0:int(W / 2)]
        real_current_v3_RU = real_current[:, :, 0:int(H / 2), int(W / 2):W]
        real_current_v3_LD = real_current[:, :, int(H / 2):H, 0:int(W / 2)]
        real_current_v3_RD = real_current[:, :, int(H / 2):H, int(W / 2):W]
        real_current_blur_v3_LU = real_current_blur[:, :, 0:int(H / 2), 0:int(W / 2)]
        real_current_blur_v3_RU = real_current_blur[:, :, 0:int(H / 2), int(W / 2):W]
        real_current_blur_v3_LD = real_current_blur[:, :, int(H / 2):H, 0:int(W / 2)]
        real_current_blur_v3_RD = real_current_blur[:, :, int(H / 2):H, int(W / 2):W]


        input_feature_v3_LU_1=self.Elayer_v3_1(self.Elayer_v3_relu(self.Elayer_v3_conv(input_current_v3_LU)))
        input_feature_v3_LU_2=self.Elayer_v3_2(input_feature_v3_LU_1)
        input_feature_v3_LU_3 = self.Elayer_v3_3(input_feature_v3_LU_2)
        #input_feature_v3_LU_4=self.Elayer_v3_4(input_feature_v3_LU_3)
        real_feature_v3_LU_1 = self.Elayer_v3_1(self.Elayer_v3_relu(self.Elayer_v3_conv(real_current_v3_LU)))
        real_feature_v3_LU_2 = self.Elayer_v3_2(real_feature_v3_LU_1)
        real_feature_v3_LU_3 = self.Elayer_v3_3(real_feature_v3_LU_2)
        #real_feature_v3_LU_4 = self.Elayer_v3_4(real_feature_v3_LU_3)
        real_blur_feature_v3_LU_1 = self.Elayer_v3_1(self.Elayer_v3_relu(self.Elayer_v3_conv(real_current_blur_v3_LU)))
        real_blur_feature_v3_LU_2 = self.Elayer_v3_2(real_blur_feature_v3_LU_1)
        real_blur_feature_v3_LU_3 = self.Elayer_v3_3(real_blur_feature_v3_LU_2)
        ranked_feature_v3_LU=self.RankingCorrelationModule(input_feature_v3_LU_3,real_feature_v3_LU_3,real_blur_feature_v3_LU_3)
        #sub_feature_v3_LU_3 = self.upsample_v3_4(self.Dlayer_v3_4(ranked_feature_v3_LU))
        sub_feature_v3_LU_2 = self.upsample_v3_3(self.Dlayer_v3_3(self.conv_trans_chnannel_v3_3(torch.cat((ranked_feature_v3_LU,input_feature_v3_LU_3),dim=1))))
        sub_feature_v3_LU_1 = self.upsample_v3_2(self.Dlayer_v3_2(self.conv_trans_chnannel_v3_2(torch.cat((sub_feature_v3_LU_2, input_feature_v3_LU_2),dim=1))))
        sub_feature_LU = self.upsample_v3_1(self.Dlayer_v3_1(self.conv_trans_chnannel_v3_1(torch.cat((sub_feature_v3_LU_1, input_feature_v3_LU_1),dim=1))))

        input_feature_v3_RU_1 = self.Elayer_v3_1(self.Elayer_v3_relu(self.Elayer_v3_conv(input_current_v3_RU)))
        input_feature_v3_RU_2 = self.Elayer_v3_2(input_feature_v3_RU_1)
        input_feature_v3_RU_3 = self.Elayer_v3_3(input_feature_v3_RU_2)
        #input_feature_v3_RU_4 = self.Elayer_v3_4(input_feature_v3_RU_3)
        real_feature_v3_RU_1 = self.Elayer_v3_1(self.Elayer_v3_relu(self.Elayer_v3_conv(real_current_v3_RU)))
        real_feature_v3_RU_2 = self.Elayer_v3_2(real_feature_v3_RU_1)
        real_feature_v3_RU_3 = self.Elayer_v3_3(real_feature_v3_RU_2)
        #real_feature_v3_RU_4 = self.Elayer_v3_4(real_feature_v3_RU_3)
        real_blur_feature_v3_RU_1 = self.Elayer_v3_1(self.Elayer_v3_relu(self.Elayer_v3_conv(real_current_blur_v3_RU)))
        real_blur_feature_v3_RU_2 = self.Elayer_v3_2(real_blur_feature_v3_RU_1)
        real_blur_feature_v3_RU_3 = self.Elayer_v3_3(real_blur_feature_v3_RU_2)
        ranked_feature_v3_RU=self.RankingCorrelationModule(input_feature_v3_RU_3,real_feature_v3_RU_3,real_blur_feature_v3_RU_3)
        #sub_feature_v3_RU_3 = self.upsample_v3_4(self.Dlayer_v3_4(ranked_feature_v3_RU))
        sub_feature_v3_RU_2 = self.upsample_v3_3(self.Dlayer_v3_3(self.conv_trans_chnannel_v3_3(torch.cat((ranked_feature_v3_RU,input_feature_v3_RU_3),dim=1))))
        sub_feature_v3_RU_1 = self.upsample_v3_2(self.Dlayer_v3_2(self.conv_trans_chnannel_v3_2(torch.cat((sub_feature_v3_RU_2, input_feature_v3_RU_2),dim=1))))
        sub_feature_RU = self.upsample_v3_1(self.Dlayer_v3_1(self.conv_trans_chnannel_v3_1(torch.cat((sub_feature_v3_RU_1, input_feature_v3_RU_1),dim=1))))

        input_feature_v3_LD_1 = self.Elayer_v3_1(self.Elayer_v3_relu(self.Elayer_v3_conv(input_current_v3_LD)))
        input_feature_v3_LD_2 = self.Elayer_v3_2(input_feature_v3_LD_1)
        input_feature_v3_LD_3 = self.Elayer_v3_3(input_feature_v3_LD_2)
        #input_feature_v3_LD_4 = self.Elayer_v3_4(input_feature_v3_LD_3)
        real_feature_v3_LD_1 = self.Elayer_v3_1(self.Elayer_v3_relu(self.Elayer_v3_conv(real_current_v3_LD)))
        real_feature_v3_LD_2 = self.Elayer_v3_2(real_feature_v3_LD_1)
        real_feature_v3_LD_3 = self.Elayer_v3_3(real_feature_v3_LD_2)
        #real_feature_v3_LD_4 = self.Elayer_v3_4(real_feature_v3_LD_3)
        real_blur_feature_v3_LD_1 = self.Elayer_v3_1(self.Elayer_v3_relu(self.Elayer_v3_conv(real_current_blur_v3_LD)))
        real_blur_feature_v3_LD_2 = self.Elayer_v3_2(real_blur_feature_v3_LD_1)
        real_blur_feature_v3_LD_3 = self.Elayer_v3_3(real_blur_feature_v3_LD_2)
        ranked_feature_v3_LD=self.RankingCorrelationModule(input_feature_v3_LD_3,real_feature_v3_LD_3,real_blur_feature_v3_LD_3)
        #sub_feature_v3_LD_3 = self.upsample_v3_4(self.Dlayer_v3_4(ranked_feature_v3_LD))
        sub_feature_v3_LD_2 = self.upsample_v3_3(self.Dlayer_v3_3(self.conv_trans_chnannel_v3_3(torch.cat((ranked_feature_v3_LD,input_feature_v3_LD_3),dim=1))))
        sub_feature_v3_LD_1 = self.upsample_v3_2(self.Dlayer_v3_2(self.conv_trans_chnannel_v3_2(torch.cat((sub_feature_v3_LD_2, input_feature_v3_LD_2),dim=1))))
        sub_feature_LD = self.upsample_v3_1(self.Dlayer_v3_1(self.conv_trans_chnannel_v3_1(torch.cat((sub_feature_v3_LD_1, input_feature_v3_LD_1),dim=1))))

        input_feature_v3_RD_1 = self.Elayer_v3_1(self.Elayer_v3_relu(self.Elayer_v3_conv(input_current_v3_RD)))
        input_feature_v3_RD_2 = self.Elayer_v3_2(input_feature_v3_RD_1)
        input_feature_v3_RD_3 = self.Elayer_v3_3(input_feature_v3_RD_2)
        #input_feature_v3_RD_4 = self.Elayer_v3_4(input_feature_v3_RD_3)
        real_feature_v3_RD_1 = self.Elayer_v3_1(self.Elayer_v3_relu(self.Elayer_v3_conv(real_current_v3_RD)))
        real_feature_v3_RD_2 = self.Elayer_v3_2(real_feature_v3_RD_1)
        real_feature_v3_RD_3 = self.Elayer_v3_3(real_feature_v3_RD_2)
        #real_feature_v3_RD_4 = self.Elayer_v3_4(real_feature_v3_RD_3)
        real_blur_feature_v3_RD_1 = self.Elayer_v3_1(self.Elayer_v3_relu(self.Elayer_v3_conv(real_current_blur_v3_RD)))
        real_blur_feature_v3_RD_2 = self.Elayer_v3_2(real_blur_feature_v3_RD_1)
        real_blur_feature_v3_RD_3 = self.Elayer_v3_3(real_blur_feature_v3_RD_2)
        ranked_feature_v3_RD=self.RankingCorrelationModule(input_feature_v3_RD_3,real_feature_v3_RD_3,real_blur_feature_v3_RD_3)
        #sub_feature_v3_RD_3 = self.upsample_v3_4(self.Dlayer_v3_4(ranked_feature_v3_RD))
        sub_feature_v3_RD_2 = self.upsample_v3_3(self.Dlayer_v3_3(self.conv_trans_chnannel_v3_3(torch.cat((ranked_feature_v3_RD,input_feature_v3_RD_3),dim=1))))
        sub_feature_v3_RD_1 = self.upsample_v3_2(self.Dlayer_v3_2(self.conv_trans_chnannel_v3_2(torch.cat((sub_feature_v3_RD_2, input_feature_v3_RD_2),dim=1))))
        sub_feature_RD = self.upsample_v3_1(self.Dlayer_v3_1(self.conv_trans_chnannel_v3_1(torch.cat((sub_feature_v3_RD_1, input_feature_v3_RD_1),dim=1))))

        sub_combined_feature_L = torch.cat((sub_feature_LU, sub_feature_LD), 2)
        sub_combined_feature_R = torch.cat((sub_feature_RU, sub_feature_RD), 2)

        input_conbined_feature_v3_L=torch.cat((ranked_feature_v3_LU,ranked_feature_v3_LD),2)
        input_conbined_feature_v3_R=torch.cat((ranked_feature_v3_RU,ranked_feature_v3_RD),2)

        input_v2_L=input_current[:,:,:,0:int(W/2)]
        input_v2_R = input_current[:, :, :, int(W / 2):W]
        real_v2_L = real_current[:, :, :, 0:int(W / 2)]
        real_v2_R = real_current[:, :, :, int(W / 2):W]
        real_blur_v2_L = real_current_blur[:, :, :, 0:int(W / 2)]
        real_blur_v2_R = real_current_blur[:, :, :, int(W / 2):W]

        input_feature_v2_L_1=self.Elayer_v2_1(self.Elayer_v2_relu(self.Elayer_v2_conv(input_v2_L+sub_combined_feature_L)))
        input_feature_v2_L_2=self.Elayer_v2_2(input_feature_v2_L_1)
        input_feature_v2_L_3=self.Elayer_v2_3(input_feature_v2_L_2)
        #input_feature_v2_L_4 = self.Elayer_v2_4(input_feature_v2_L_3)
        real_feature_v2_L_1 = self.Elayer_v2_1(self.Elayer_v2_relu(self.Elayer_v2_conv(real_v2_L)))
        real_feature_v2_L_2 = self.Elayer_v2_2(real_feature_v2_L_1)
        real_feature_v2_L_3 = self.Elayer_v2_3(real_feature_v2_L_2)
        #real_feature_v2_L_4 = self.Elayer_v2_4(real_feature_v2_L_3)
        real_blur_feature_v2_L_1 = self.Elayer_v2_1(self.Elayer_v2_relu(self.Elayer_v2_conv(real_blur_v2_L)))
        real_blur_feature_v2_L_2 = self.Elayer_v2_2(real_blur_feature_v2_L_1)
        real_blur_feature_v2_L_3 = self.Elayer_v2_3(real_blur_feature_v2_L_2)
        ranked_feature_v2_L=self.RankingCorrelationModule(input_feature_v2_L_3,real_feature_v2_L_3,real_blur_feature_v2_L_3)
        #sub_feature_v2_L_3=self.upsample_v2_4(self.Dlayer_v2_4(ranked_feature_v2_L+input_conbined_feature_v3_L))
        sub_feature_v2_L_2=self.upsample_v2_3(self.Dlayer_v2_3(self.conv_trans_chnannel_v2_3(torch.cat((ranked_feature_v2_L,input_conbined_feature_v3_L,input_feature_v2_L_3),dim=1))))
        sub_feature_v2_L_1 = self.upsample_v2_2(self.Dlayer_v2_2(self.conv_trans_chnannel_v2_2(torch.cat((sub_feature_v2_L_2, input_feature_v2_L_2),dim=1))))
        sub_feature_L = self.upsample_v2_1(self.Dlayer_v2_1(self.conv_trans_chnannel_v2_1(torch.cat((sub_feature_v2_L_1, input_feature_v2_L_1),dim=1))))

        input_feature_v2_R_1 = self.Elayer_v2_1(self.Elayer_v2_relu(self.Elayer_v2_conv(input_v2_R+sub_combined_feature_R)))
        input_feature_v2_R_2 = self.Elayer_v2_2(input_feature_v2_R_1)
        input_feature_v2_R_3 = self.Elayer_v2_3(input_feature_v2_R_2)
        #input_feature_v2_R_4 = self.Elayer_v2_4(input_feature_v2_R_3)
        real_feature_v2_R_1 = self.Elayer_v2_1(self.Elayer_v2_relu(self.Elayer_v2_conv(real_v2_R)))
        real_feature_v2_R_2 = self.Elayer_v2_2(real_feature_v2_R_1)
        real_feature_v2_R_3 = self.Elayer_v2_3(real_feature_v2_R_2)
        #real_feature_v2_R_4 = self.Elayer_v2_4(real_feature_v2_R_3)
        real_blur_feature_v2_R_1 = self.Elayer_v2_1(self.Elayer_v2_relu(self.Elayer_v2_conv(real_blur_v2_R)))
        real_blur_feature_v2_R_2 = self.Elayer_v2_2(real_blur_feature_v2_R_1)
        real_blur_feature_v2_R_3 = self.Elayer_v2_3(real_blur_feature_v2_R_2)
        ranked_feature_v2_R=self.RankingCorrelationModule(input_feature_v2_R_3,real_feature_v2_R_3,real_blur_feature_v2_R_3)
        #sub_feature_v2_R_3 = self.upsample_v2_4(self.Dlayer_v2_4(ranked_feature_v2_R+input_conbined_feature_v3_R))
        sub_feature_v2_R_2 = self.upsample_v2_3(self.Dlayer_v2_3(self.conv_trans_chnannel_v2_3(torch.cat((ranked_feature_v2_R,input_conbined_feature_v3_R,input_feature_v2_R_3),dim=1))))
        sub_feature_v2_R_1 = self.upsample_v2_2(self.Dlayer_v2_2(self.conv_trans_chnannel_v2_2(torch.cat((sub_feature_v2_R_2, input_feature_v2_R_2),dim=1))))
        sub_feature_R = self.upsample_v2_1(self.Dlayer_v2_1(self.conv_trans_chnannel_v2_1(torch.cat((sub_feature_v2_R_1, input_feature_v2_R_1),dim=1))))

        sub_feature=torch.cat((sub_feature_L, sub_feature_R), 3)
        input_conbined_feature_v2=torch.cat((ranked_feature_v2_L,ranked_feature_v2_R),3)

        input_feature_v1_1=self.Elayer_v1_1(self.Elayer_v1_relu(self.Elayer_v1_conv(sub_feature+input_current)))
        input_feature_v1_2=self.Elayer_v1_2(input_feature_v1_1)
        input_feature_v1_3 = self.Elayer_v1_3(input_feature_v1_2)
        #input_feature_v1_4 = self.Elayer_v1_4(input_feature_v1_3)
        real_feature_v1_1 = self.Elayer_v1_1(self.Elayer_v1_relu(self.Elayer_v1_conv(real_current)))
        real_feature_v1_2 = self.Elayer_v1_2(real_feature_v1_1)
        real_feature_v1_3 = self.Elayer_v1_3(real_feature_v1_2)
        #real_feature_v1_4 = self.Elayer_v1_4(real_feature_v1_3)
        real_blur_feature_v1_1 = self.Elayer_v1_1(self.Elayer_v1_relu(self.Elayer_v1_conv(real_current_blur)))
        real_blur_feature_v1_2 = self.Elayer_v1_2(real_blur_feature_v1_1)
        real_blur_feature_v1_3 = self.Elayer_v1_3(real_blur_feature_v1_2)
        ranked_feature_v1=self.RankingCorrelationModule(input_feature_v1_3,real_feature_v1_3,real_blur_feature_v1_3)
        #sub_feature_v1_3 = self.upsample_v1_4(self.Dlayer_v1_4(ranked_feature_v1+input_conbined_feature_v2))
        sub_feature_v1_2 = self.upsample_v1_3(self.Dlayer_v1_3(self.conv_trans_chnannel_v1_3(torch.cat((ranked_feature_v1,input_conbined_feature_v2,input_feature_v1_3),dim=1))))
        sub_feature_v1_1 = self.upsample_v1_2(self.Dlayer_v1_2(self.conv_trans_chnannel_v1_2(torch.cat((sub_feature_v1_2, input_feature_v1_2),dim=1))))
        out = self.upsample_v1_1(self.Dlayer_v1_1(self.conv_trans_chnannel_v1_1(torch.cat((sub_feature_v1_1, input_feature_v1_1),dim=1))))

        return self.tanh(out)