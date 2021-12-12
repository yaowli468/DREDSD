
def create_model(opt):
	model = None
	if opt.model == 'test':
		assert (opt.dataset_mode == 'single')
		from .multi_test_model import TestModel
		model = TestModel( opt )

	# model.initialize(opt)
	print("model [%s] was created" % (model.name()))
	return model
