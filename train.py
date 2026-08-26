import os
import time
import inspect
import random

from termcolor import colored, cprint
from tqdm import tqdm

import torch.backends.cudnn as cudnn
# cudnn.benchmark = True

from options.train_options import TrainOptions
from datasets.dataloader import config_dataloader, get_data_generator
from models.base_model import create_model

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_descriptor')

from utils.distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)

from utils.util import convert_partial_pcd_2_mesh, seed_everything, category_2_to_label

import torch
from utils.visualizer import Visualizer

import numpy as np

from utils.util import Projection_List, Projection_List_zero
from omegaconf import OmegaConf

def train_main_worker(opt, model, train_loader, test_loader, validation_loader, visualizer):

    if get_rank() == 0:
        cprint('[*] Start training. name: %s' % opt.name, 'blue')

    train_dg = get_data_generator(train_loader)
    test_dg = get_data_generator(test_loader) if test_loader is not None else None

    epoch_length = len(train_loader)
    print('The epoch length is', epoch_length)

    start_iter = opt.start_iter
    additional_iters = epoch_length * opt.epochs
    total_iters = start_iter + additional_iters

    epoch = start_iter // epoch_length

    # pbar = tqdm(total=total_iters)
    pbar = tqdm(range(start_iter, total_iters))

    best_validation_loss = float("inf")

    iter_start_time = time.time()
    for iter_i in range(start_iter, total_iters):
        opt.iter_i = iter_i
        iter_ip1 = iter_i + 1

        if get_rank() == 0:
            visualizer.reset()

        data = next(train_dg) # sdf samples and pointcloud
        data['iter_num'] = iter_i
        data['epoch'] = epoch
        model.set_input(data)
        model.optimize_parameters()

        # if torch.isnan(model.loss).any() == True:
        #     break

        if get_rank() == 0:
            pbar.update(1)
            if iter_i % opt.print_freq == 0:
                errors = model.get_current_errors()

                t = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_errors(iter_i, errors, t)

            if iter_ip1 % opt.save_latest_freq == 0:
                cprint('saving the latest model (current_iter %d)' % (iter_i), 'blue')
                latest_name = f'steps-latest'
                model.save(latest_name, iter_ip1)

            # save every 3000 steps (batches)
            if iter_ip1 % opt.save_steps_freq == 0:
                cprint('saving the model at iters %d' % iter_ip1, 'blue')
                latest_name = f'steps-latest'
                model.save(latest_name, iter_ip1)
                cur_name = f'steps-{iter_ip1}'
                model.save(cur_name, iter_ip1)

                cprint(f'[*] End of steps %d \t Time Taken: %d sec \n%s' %
                    (
                        iter_ip1,
                        time.time() - iter_start_time,
                        os.path.abspath(os.path.join(opt.logs_dir, opt.name))
                    ), 'blue', attrs=['bold']
                )

            if iter_i % epoch_length == epoch_length - 1:
                print('Finish One Epoch!')
                epoch += 1
                print('Now Epoch is:', epoch)

                if get_rank() == 0 and validation_loader is not None:
                    validation_loss = evaluate_validation(model, validation_loader)
                    print(f'Validation Loss: {validation_loss:.6f}')

                    if validation_loss < best_validation_loss:
                        best_validation_loss = validation_loss
                        print(f'New best model! Saving with validation_loss={validation_loss:.6f}')
                        model.save('best', iter_ip1)

        # display every n batches
        if iter_i % opt.display_freq == 0:
            if iter_i == 0 and opt.debug == "0":
                pbar.update(1)
                continue

            # eval
            if opt.model == "vae":
                if test_dg is not None:
                    data = next(test_dg)
                    data['iter_num'] = iter_i
                    data['epoch'] = epoch
                    model.set_input(data)
                    model.inference(save_folder = f'temp/{iter_i}')
            else:
                if opt.category == "im_2":
                    category = random.choice(list(category_2_to_label.keys()))
                else:
                    category = opt.category
                
                model.sample(category = category, prefix = 'results', ema = True, ddim_steps = 200, save_index = iter_i)
            
            # torch.cuda.empty_cache()

        if opt.update_learning_rate:
            model.update_learning_rate_cos(epoch, opt)

def evaluate_validation(model, validation_loader):
    model.eval()

    validation_dg = get_data_generator(validation_loader)
    num_batches = len(validation_loader)

    total_loss = 0.0

    with torch.no_grad():
        for _ in range(num_batches):
            data = next(validation_dg)
            model.set_input(data)

            loss = model.compute_validation_loss()
            total_loss += loss.item()

    model.train()

    return total_loss / num_batches

def generate_vae(opt, model, test_loader):
    if get_rank() == 0:
        cprint('[*] Start training. name: %s' % opt.name, 'blue')

    test_dg = get_data_generator(test_loader)

    epoch_length = len(train_loader)
    print('The epoch length is', epoch_length)

    total_iters = epoch_length
    start_iter = 0

    # pbar = tqdm(total=total_iters)
    pbar = tqdm(range(start_iter, total_iters))

    for iter_i in range(start_iter, total_iters):

        data = next(test_dg)
        data['iter_num'] = iter_i
        data['epoch'] = 0
        model.set_input(data)
        seed_everything(opt.seed)
        model.inference()
        #model.inference_shape_completion()
        pbar.update

""" TODO adapt to generate many views for generation metrics calculation
def generate(opt, model):
    # get n_epochs here
    total_iters = 100000000
    pbar = tqdm(total=total_iters)

    total_num = category_5_to_num[opt.category]

    for iter_i in range(total_iters):

        result_index = iter_i * get_world_size() + get_rank()
        if opt.split_dir is not None:
            split_path = os.path.join(opt.split_dir, f'{result_index}.pth')
            split_small = torch.load(split_path)
            split_small = split_small.to(model.device)
        else:
            split_small = None
        model.batch_size = 1
        
        if result_index >= total_num: 
            break

        if opt.category == "im_2":
            category = random.choice(list(category_2_to_label.keys()))
        else:
            category = opt.category
        model.sample(split_small = split_small, category = category, prefix = 'results', ema = True, ddim_steps = 200, clean = False, save_index = result_index)
        pbar.update(1)
"""

def generate(opt, model, loader):
    generator = get_data_generator(loader)

    epoch_length = len(loader)
    total_iters = epoch_length
    start_iter = 0

    pbar = tqdm(range(start_iter, total_iters))
    for i in range(start_iter, total_iters):
        result_index = i * get_world_size() + get_rank()
        data = next(generator)

        if model.use_sketch_condition:
            diffusion_conf = OmegaConf.load(opt.df_cfg)

            view = 0
            sample = 0
            image_feature = np.load(os.path.join(data['feature_folder'][0], f"{view}{sample}.npy"))
            image_feature = image_feature[1:]
            image_feature = torch.from_numpy(image_feature).float()
            image_feature = image_feature.unsqueeze(0)
            data['img_condition'] = image_feature
            data['edge_path'] = os.path.join(data['edge_path'][0], f"edge_{view}_{sample}.png")

            if diffusion_conf.unet.params.elevation_zero:
                pm = Projection_List_zero[view]
            else:
                pm = Projection_List[view]

            projection_matrix = np.expand_dims(pm, 0)
            projection_matrix = torch.from_numpy(projection_matrix).float()
            projection_matrix = projection_matrix.unsqueeze(0)
            data['projection_matrix'] = projection_matrix

        if opt.category == "im_2":
            category = random.choice(list(category_2_to_label.keys()))
        else:
            category = opt.category

        model.set_input(data)
        model.sample(ema = True, ddim_steps = 200, save_index = result_index, is_training=False, category=category)
        pbar.update()

if __name__ == "__main__":
    # this will parse args, setup log_dirs, multi-gpus
    opt = TrainOptions().parse_and_setup()
    device = opt.device
    rank = opt.rank

    # CUDA_VISIBLE_DEVICES = int(os.environ["LOCAL_RANK"])
    # import pdb; pdb.set_trace()

    # get current time, print at terminal. easier to track exp
    from datetime import datetime
    opt.exp_time = datetime.now().strftime('%Y-%m-%dT%H-%M')      

    # main loop
    model = create_model(opt)
    opt.start_iter = model.start_iter
    cprint(f'[*] "{opt.model}" initialized.', 'cyan')

    # visualizer
    visualizer = Visualizer(opt)
    if get_rank() == 0:
        visualizer.setup_io()

    # save model and dataset files
    if get_rank() == 0:
        expr_dir = '%s/%s' % (opt.logs_dir, opt.name)
        model_f = inspect.getfile(model.__class__)
        modelf_out = os.path.join(expr_dir, os.path.basename(model_f))
        os.system(f'cp {model_f} {modelf_out}')
        if opt.model != "vae":
            unet_f = inspect.getfile(model.df_module.__class__)
            unetf_out = os.path.join(expr_dir, os.path.basename(unet_f))
            os.system(f'cp {unet_f} {unetf_out}')
        dset_f = "datasets/dualoctree_snet.py"
        dsetf_out = os.path.join(expr_dir, os.path.basename(dset_f))
        os.system(f'cp {dset_f} {dsetf_out}')
        train_f = 'train.py'
        train_out = os.path.join(expr_dir, os.path.basename(train_f))
        os.system(f'cp {train_f} {train_out}')
        
        if opt.vq_cfg is not None:
            vq_cfg = opt.vq_cfg
            cfg_out = os.path.join(expr_dir, os.path.basename(vq_cfg))
            os.system(f'cp {vq_cfg} {cfg_out}')

        if opt.df_cfg is not None:
            df_cfg = opt.df_cfg
            cfg_out = os.path.join(expr_dir, os.path.basename(df_cfg))
            os.system(f'cp {df_cfg} {cfg_out}')
    if opt.mode == 'train':
        # if opt.debug == "0":
        #     # try:
        #     #     train_main_worker(opt, model, train_loader, test_loader, visualizer)
        #     # except:
        #     #     import traceback
        #     #     print(traceback.format_exc(), flush=True)
        #     #     with open(os.path.join(opt.logs_dir, opt.name, "error.txt"), "a") as f:
        #     #         f.write(traceback.format_exc() + "\n")
        #     #     raise ValueError
        # else:
        train_loader, test_loader, validation_loader = config_dataloader(opt)
        train_main_worker(opt, model, train_loader, test_loader, validation_loader, visualizer)
    elif opt.mode == 'generate':
        train_loader, test_loader, validation_loader = config_dataloader(opt)
        if opt.model == "vae":
            generate_vae(opt, model, test_loader)
        else:
            # generate(opt, model, train_loader)
            generate(opt, model, test_loader)
    else:
        raise ValueError


