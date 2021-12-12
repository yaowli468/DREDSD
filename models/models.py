from .train_model import ConditionalGAN

def create_model(opt):
	if opt.model == 'test':
		assert (opt.dataset_mode == 'single')
		from .test_model import TestModel
		model = TestModel( opt )
	else:
		model = ConditionalGAN(opt)
	print("model [%s] was created" % (model.name()))
	return model
