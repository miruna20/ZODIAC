import os
import torch.utils.data

from datasets.sampler import InfSampler, DistributedInfSampler
from builder import get_dataset
from omegaconf import OmegaConf
from termcolor import colored, cprint

def get_data_generator(loader):
	while True:
		for data in loader:
			yield data

def config_dataloader(opt):
	dualoctree_conf = OmegaConf.load(opt.vq_cfg)
	diffusion_conf = OmegaConf.load(opt.df_cfg)
	flags_train, flags_test, flags_validation = dualoctree_conf.data.train, dualoctree_conf.data.test, dualoctree_conf.data.validation
	extra_flags = diffusion_conf.unet.params

	if "spines" in opt.vq_cfg:
		flags_train.filelist = os.path.join(flags_train.filelist, f'train.txt')
		flags_test.filelist = os.path.join(flags_test.filelist, f'test.txt')
		flags_validation.filelist = os.path.join(flags_validation.filelist, f'validation.txt')
	elif "snet" in opt.vq_cfg:
		flags_train.filelist = os.path.join(flags_train.filelist, f'train_{opt.category}.txt')
		flags_test.filelist = os.path.join(flags_test.filelist, f'test_{opt.category}.txt')
	else:
		raise Exception("Dateset unknown. Only configured for ShapeNet and spines")

	train_loader = None
	if not flags_train.disable:
		train_loader = get_dataloader(opt, flags_train, drop_last = False, extra_flags=extra_flags)
		train_ds = train_loader.dataset
		cprint('[*] # training images = %d' % len(train_ds), 'yellow')

	test_loader = None
	if not flags_test.disable:
		test_loader = get_dataloader(opt, flags_test, drop_last = False, extra_flags=extra_flags)
		test_ds = test_loader.dataset
		cprint('[*] # testing images = %d' % len(test_ds), 'yellow')
	
	validation_loader = None
	if not flags_validation.disable:
		validation_loader = get_dataloader(opt, flags_validation, drop_last = False, extra_flags=extra_flags)
		validation_ds = validation_loader.dataset
		cprint('[*] # validation images = %d' % len(validation_ds), 'yellow')

	return train_loader, test_loader, validation_loader

def get_dataloader(opt, flags, drop_last = False, extra_flags=None):
	dataset, collate_fn = get_dataset(flags, extra_flags)

	if opt.distributed:
		sampler = DistributedInfSampler(dataset, shuffle=flags.shuffle)
	else:
		sampler = InfSampler(dataset, shuffle=flags.shuffle)

	data_loader = torch.utils.data.DataLoader(
		dataset, batch_size=flags.batch_size, num_workers=flags.num_workers,
		sampler=sampler, collate_fn=collate_fn, pin_memory=True, drop_last = drop_last)

	return data_loader
