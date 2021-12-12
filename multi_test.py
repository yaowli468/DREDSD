import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.multi_models import create_model
from util.visualizer import Visualizer
from util import html

# if public_dataset: python multi_test.py --dataroot=./testSet/testImage --exemplar_dir=./testSet/exemplar_AB
# if real_dataset: python multi_test.py --no_blur_exemplar --dataroot=./testSet/testImage --exemplar_dir=./testSet/exemplar_AB

opt = TestOptions().parse()
opt.nThreads = 1
opt.batchSize = 1
opt.serial_batches = True
opt.no_flip = True

opt.dataset_mode='single'
opt.model='test'
opt.resize_or_crop='none'

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
# test

counter = 0
for i, data in enumerate(dataset):
	if i >= opt.how_many:
		break
	counter = i
	model.set_input(data)
	model.test()
	visuals = model.get_current_visuals()
	img_path = model.get_image_paths()
	print('process image... %s' % img_path)
	visualizer.save_images(webpage, visuals, img_path)

webpage.save()
