import numpy as np
import torch
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from .losses import init_loss

try:
	xrange          # Python2
except NameError:
	xrange = range  # Python 3

class ConditionalGAN(BaseModel):
	def name(self):
		return 'ConditionalGANModel'

	def __init__(self, opt):
		super(ConditionalGAN, self).__init__(opt)
		self.isTrain = opt.isTrain
		self.netG_name=opt.which_model_netG
		# define tensors
		self.input_A = self.Tensor(opt.batchSize, opt.input_nc,  opt.fineWidthSize, opt.fineheightSize)
		self.input_B = self.Tensor(opt.batchSize, opt.output_nc, opt.fineWidthSize, opt.fineheightSize)
		self.input_E = self.Tensor(opt.batchSize, opt.output_nc, opt.fineWidthSize, opt.fineheightSize)
		self.input_E_blur = self.Tensor(opt.batchSize, opt.output_nc, opt.fineWidthSize, opt.fineheightSize)

		use_parallel = not opt.gan_type == 'wgan-gp'
		print("Use Parallel = ", "True" if use_parallel else "False")
		self.netG = networks.define_G(opt.which_model_netG, self.gpu_ids)

		if self.isTrain:
		 	use_sigmoid = opt.gan_type == 'gan'
		 	self.netD = networks.define_D(
		 		opt.output_nc, opt.ndf, opt.which_model_netD,
		 		opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids, use_parallel
		 	)

		if self.isTrain:
			self.fake_AB_pool = ImagePool(opt.pool_size)

			# initialize optimizers
			self.optimizer_G = torch.optim.Adam( self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999) )
			self.optimizer_D = torch.optim.Adam( self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999) )
			self.optimizers.append(self.optimizer_G)
			self.optimizers.append(self.optimizer_D)
												
			self.criticUpdates = 1 #if opt.gan_type == 'wgan-gp' else 1
			
			# define loss functions
			self.discLoss, self.perceptualLoss, self.contentLoss,self.ssim_loss = init_loss(opt, self.Tensor)

		print('---------- Networks initialized -------------')
		networks.print_network(self.netG)
		if self.isTrain:
			networks.print_network(self.netG)
		print('-----------------------------------------------')

	def set_input(self, input):
		AtoB = self.opt.which_direction == 'AtoB'
		inputA = input['A' if AtoB else 'B']
		inputB = input['B' if AtoB else 'A']
		inputE = input['E' if AtoB else 'A']
		inputE_blur=input['E_blur']
		self.input_A.resize_(inputA.size()).copy_(inputA)
		self.input_B.resize_(inputB.size()).copy_(inputB)
		self.input_E.resize_(inputE.size()).copy_(inputE)
		self.input_E_blur.resize_(inputE_blur.size()).copy_(inputE_blur)
		self.image_paths = input['A_paths' if AtoB else 'B_paths']

	def forward(self):
		self.real_A = Variable(self.input_A)
		self.real_E=Variable(self.input_E)
		self.real_B = Variable(self.input_B)
		self.real_E_blur=Variable(self.input_E_blur)
		self.fake_B = self.netG.forward(self.real_A, self.real_E,self.real_E_blur)

	# no backprop gradients
	def test(self):
		self.real_A = Variable(self.input_A, volatile=True)
		self.fake_B = self.netG.forward(self.real_A)
		self.real_B = Variable(self.input_B, volatile=True)

	# get image paths
	def get_image_paths(self):
		return self.image_paths

	def backward_D(self):
	 	self.loss_D = self.discLoss.get_loss(self.netD, self.fake_B, self.real_B)
	
	 	self.loss_D.backward(retain_graph=True)

	def backward_G(self):
		self.loss_G_GAN = self.discLoss.get_g_loss(self.netD, self.fake_B, self.real_B)*0.1

		mask = self.hard_mining_mask(self.fake_B, self.real_B).detach()
		self.loss_G_Content=self.contentLoss.get_loss(self.fake_B * mask,self.real_B *mask)*500

		self.loss_G = self.loss_G_Content+self.loss_G_GAN

		self.loss_G.backward()

	def optimize_parameters(self):
		self.forward()

		for iter_d in xrange(self.criticUpdates):
		 	self.optimizer_D.zero_grad()
		 	self.backward_D()
		 	self.optimizer_D.step()

		self.optimizer_G.zero_grad()
		self.backward_G()
		self.optimizer_G.step()

	def get_current_errors(self):
		return OrderedDict([('G_GAN', self.loss_G_GAN.item()),
							('G_con', self.loss_G_Content.item()),
							('D_real+fake', self.loss_D.item())
							])

	def get_current_visuals(self):
		real_A = util.tensor2im(self.real_A.data)
		fake_B = util.tensor2im(self.fake_B.data)
		real_B = util.tensor2im(self.real_B.data)
		return OrderedDict([('Blurred_Train', real_A), ('Restored_Train', fake_B), ('Sharp_Train', real_B)])

	def save(self, label):
		self.save_network(self.netG, 'G', label, self.gpu_ids)
		self.save_network(self.netD, 'D', label, self.gpu_ids)

	def load_network(self, load_path, network, strict=True):
		if isinstance(network, torch.nn.DataParallel):
			network = network.module
		load_net = torch.load(load_path)
		load_net_clean = OrderedDict()  # remove unnecessary 'module.'
		for k, v in load_net.items():
			if k.startswith('module.'):
				load_net_clean[k[7:]] = v
			else:
				load_net_clean[k] = v
		network.load_state_dict(load_net_clean, strict=strict)

	def hard_mining_mask(self, x, y):
		x = x.detach()
		y = y.detach()
		with torch.no_grad():
			b, c, h, w = x.size()

			hard_mask = np.zeros(shape=(b, 1, h, w))
			res = torch.sum(torch.abs(x - y), dim=1, keepdim=True)
			res_numpy = res.cpu().numpy()
			res_line = res.view(b, -1)
			res_sort = [res_line[i].sort(descending=True) for i in range(b)]
			hard_thre_ind = int(0.5 * w * h)
			for i in range(b):
				thre_res = res_sort[i][0][hard_thre_ind].item()
				hard_mask[i] = (res_numpy[i] > thre_res).astype(np.float32)

			random_thre_ind = int(0.1 * w * h)
			random_mask = np.zeros(shape=(b, 1 * h * w))
			for i in range(b):
				random_mask[i, :random_thre_ind] = 1.
				np.random.shuffle(random_mask[i])
			random_mask = np.reshape(random_mask, (b, 1, h, w))

			mask = hard_mask + random_mask
			mask = (mask > 0.).astype(np.float32)

			mask = torch.Tensor(mask).to('cuda')

		return mask
