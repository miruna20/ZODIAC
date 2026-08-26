# Reference: diffusion is borrowed from the LDM repo: https://github.com/CompVis/latent-diffusion
# Specifically, functions from: https://github.com/CompVis/latent-diffusion/blob/main/ldm/models/diffusion/ddpm.py

import os
from collections import OrderedDict
from functools import partial
import copy
import numpy as np
from omegaconf import OmegaConf
from termcolor import colored, cprint
from einops import rearrange, repeat
from tqdm import tqdm
from random import random
import ocnn
from ocnn.nn import octree2voxel, octree_pad
from ocnn.octree import Octree, Points
from models.networks.dualoctree_networks import dual_octree
import open3d as o3d
import h5py
import time
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.special import expm1

from models.base_model import BaseModel
from models.networks.diffusion_networks.graph_unet_union import UNet3DModel
from models.model_utils import load_dualoctree
from models.networks.diffusion_networks.ldm_diffusion_util import *

# distributed
from utils.distributed import reduce_loss_dict, get_rank, get_world_size

# rendering
from utils.util_dualoctree import calc_sdf, octree2split_small, octree2split_large, split2octree_small, split2octree_large
from utils.util import TorchRecoder, seed_everything, category_2_to_label, render_mask_only, compute_visible_connections, render_voxel_scene

from PIL import Image

TRUNCATED_TIME = 0.7


class OctFusionModel(BaseModel):
    def name(self):
        return 'SDFusion-Model-Union-Two-Times'

    def initialize(self, opt):
        self.network_initialize(opt)
        self.optimizer_initialize(opt)
        
    def network_initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.mode == "train"
        self.model_name = self.name()
        self.device = opt.device
        self.gradient_clip_val = 1.
        self.start_iter = opt.start_iter

        if self.isTrain:
            self.log_dir = os.path.join(opt.logs_dir, opt.name)
            self.train_dir = os.path.join(self.log_dir, 'train_temp')
            self.test_dir = os.path.join(self.log_dir, 'test_temp')


        ######## START: Define Networks ########
        assert opt.df_cfg is not None
        assert opt.vq_cfg is not None

        # init df
        df_conf = OmegaConf.load(opt.df_cfg)
        vq_conf = OmegaConf.load(opt.vq_cfg)
        self.batch_size = vq_conf.data.train.batch_size = opt.batch_size
        
        self.df_conf = df_conf
        self.vq_conf = vq_conf
        self.solver = self.vq_conf.solver

        self.input_depth = self.vq_conf.model.depth
        self.octree_depth = self.vq_conf.model.depth_stop
        self.small_depth = 6
        self.large_depth = 8
        self.full_depth = self.vq_conf.model.full_depth

        self.load_octree = self.vq_conf.data.train.load_octree
        self.load_pointcloud = self.vq_conf.data.train.load_pointcloud
        self.load_split_small = self.vq_conf.data.train.load_split_small
        self.load_partial = self.vq_conf.data.train.load_partial

        self.use_sketch_condition = self.df_conf.unet.params.get("use_sketch_condition", False)
        self.use_partial_pcd_condition = self.df_conf.unet.params.use_partial_pcd_condition
        self.use_partial_pcd_zero_shot = self.df_conf.unet.params.use_partial_pcd_zero_shot
        self.use_partial_pcd_zero_shot_clean = self.df_conf.unet.params.use_partial_pcd_zero_shot_clean

        self.partial_cond_weight = self.df_conf.unet.params.partial_cond_weight

        # init diffusion networks
        df_model_params = df_conf.model.params
        unet_params = df_conf.unet.params
        self.conditioning_key = df_model_params.conditioning_key
        self.num_timesteps = df_model_params.timesteps
        self.enable_label = "num_classes" in df_conf.unet.params
        self.df_type = unet_params.df_type

        self.df = UNet3DModel(opt.stage_flag, **unet_params)
        self.df.to(self.device)
        self.stage_flag = opt.stage_flag

        # record z_shape
        self.split_channel = 8
        self.code_channel = self.vq_conf.model.embed_dim
        z_sp_dim = 2 ** self.full_depth
        self.z_shape = (self.split_channel, z_sp_dim, z_sp_dim, z_sp_dim)

        self.ema_df = copy.deepcopy(self.df)
        self.ema_df.to(self.device)
        if opt.isTrain:
            self.ema_rate = opt.ema_rate
            self.ema_updater = EMA(self.ema_rate)
            self.reset_parameters()
            set_requires_grad(self.ema_df, False)

        self.noise_schedule = "linear"
        if self.noise_schedule == "linear":
            self.log_snr = beta_linear_log_snr
        elif self.noise_schedule == "cosine":
            self.log_snr = alpha_cosine_log_snr
        else:
            raise ValueError(f'invalid noise schedule {self.noise_schedule}')

        # init vqvae

        self.autoencoder = load_dualoctree(conf = vq_conf, ckpt = opt.vq_ckpt, opt = opt)

        ######## END: Define Networks ########

    def optimizer_initialize(self, opt):
        if opt.pretrain_ckpt is not None:
            self.load_ckpt(opt.pretrain_ckpt, self.df, self.ema_df, load_options=["unet_lr"])
        
        if self.stage_flag == "lr":
            self.set_requires_grad([
                self.df.unet_hr
            ], False)
        elif self.stage_flag == "hr":
            self.set_requires_grad([
                self.df.unet_lr
            ], False)
        
        if self.isTrain:

            # initialize optimizers
            self.optimizer = optim.AdamW([p for p in self.df.parameters() if p.requires_grad == True], lr=opt.lr)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 1000, 0.9)

            self.optimizers = [self.optimizer]
            self.schedulers = [self.scheduler]

            self.print_networks(verbose=False)

        if opt.ckpt is None and os.path.exists(os.path.join(opt.logs_dir, opt.name, "ckpt/df_steps-latest.pth")):
            opt.ckpt = os.path.join(opt.logs_dir, opt.name, "ckpt/df_steps-latest.pth")
        
        if opt.ckpt is not None:
            if self.stage_flag == "lr":
                load_options = ["unet_lr"]
            elif self.stage_flag == "hr":
                load_options = ["unet_lr", "unet_hr"] # use load_options = ["unet_hr"] if you want to use checkpoint from 'pretain_ckpt' for LR and checkpoint from 'ckpt' for HR. See .sh file
            if self.isTrain:
                load_options.append("opt")
            self.load_ckpt(opt.ckpt, self.df, self.ema_df, load_options)
                
        trainable_params_num = 0
        for m in [self.df]:
            trainable_params_num += sum([p.numel() for p in m.parameters() if p.requires_grad == True])
        print("Trainable_params: ", trainable_params_num)

        # for distributed training
        if self.opt.distributed:
            self.make_distributed(opt)
            self.df_module = self.df.module
            self.autoencoder_module = self.autoencoder.module

        else:
            self.df_module = self.df
            self.autoencoder_module = self.autoencoder

    def reset_parameters(self):
        self.ema_df.load_state_dict(self.df.state_dict())

    def make_distributed(self, opt):
        self.df = nn.parallel.DistributedDataParallel(
            self.df,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
        if opt.sync_bn:
            self.autoencoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.autoencoder)
        self.autoencoder = nn.parallel.DistributedDataParallel(
            self.autoencoder,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True
        )

    ############################ START: init diffusion params ############################

    def batch_to_cuda(self, batch):
        def points2octree(points):
            octree = ocnn.octree.Octree(depth = self.input_depth, full_depth = self.full_depth)
            octree.build_octree(points)
            return octree

        if self.load_pointcloud:
            points = [pts.cuda(non_blocking=True) for pts in batch['points']]
            octrees = [points2octree(pts) for pts in points]
            octree = ocnn.octree.merge_octrees(octrees)
            octree.construct_all_neigh()
            batch['octree_in'] = octree
            batch['split_small'] = octree2split_small(batch['octree_in'], self.full_depth)

        if self.load_partial:
            partial_points = [pts.cuda(non_blocking=True) for pts in batch['partial_points']]
            partial_octrees = [points2octree(pts) for pts in partial_points]
            partial_octree = ocnn.octree.merge_octrees(partial_octrees)
            partial_octree.construct_all_neigh()
            batch['octree_in_partial'] = partial_octree
            batch['split_small_partial'] = octree2split_small(batch['octree_in_partial'], self.full_depth)

        batch['label'] = batch['label'].cuda()
        if self.load_octree:
            batch['octree_in'] = batch['octree_in'].cuda()
            batch['split_small'] = octree2split_small(batch['octree_in'], self.full_depth)
            # batch['split_large'] = self.octree2split_large(batch['octree_in'])
        elif self.load_split_small:
            batch['split_small'] = batch['split_small'].cuda()
            batch['octree_in'] = split2octree_small(batch['split_small'], self.input_depth, self.full_depth)

    def set_input(self, input=None):
        self.batch_to_cuda(input)
        self.split_small = input['split_small']
        self.filename = input['filename'] # this is a batch of filenames ['filename1', 'filename2']

        if self.load_partial:
            self.split_small_partial = input['split_small_partial']
            self.partial_points = input['partial_points']
        else:
            self.split_small_partial = None

        # self.split_large = input['split_large']
        self.octree_in = input['octree_in']
        self.batch_size = self.octree_in.batch_size

        if self.enable_label:
            self.label = input['label']
        else:
            self.label = None

        if self.use_sketch_condition:
            self.img_condition = input['img_condition'].to(self.opt.device)
            self.projection_matrix = input['projection_matrix'].to(self.opt.device).to(dtype=torch.float64)
            self.kernel_size = self.df_conf.unet.params.kernel_size
            self.edge_path = input['edge_path']
        else:
            self.img_condition = None
            self.projection_matrix = None
            self.kernel_size = None

    def switch_train(self):
        self.df.train()

    def switch_eval(self):
        self.df.eval()

    def calc_loss(self, input_data, doctree_in, batch_id, unet_type, unet_lr, df_type="x0"):
        times = torch.zeros(
            (self.batch_size,), device=self.device).float().uniform_(0, 1)
        
        noise = torch.randn_like(input_data)

        noise_level = self.log_snr(times)
        alpha, sigma = log_snr_to_alpha_sigma(noise_level)
        batch_alpha = right_pad_dims_to(input_data, alpha[batch_id])
        batch_sigma = right_pad_dims_to(input_data, sigma[batch_id])
        noised_data = batch_alpha * input_data + batch_sigma * noise

        x_partial_cond = None
        activate_partial_pcd_condition = self.use_partial_pcd_condition and unet_type == "lr"
        if activate_partial_pcd_condition:
            x_partial_cond = self.split_small_partial

        output = self.df(unet_type=unet_type, x=noised_data, doctree=doctree_in, unet_lr=unet_lr, timesteps=noise_level, label=self.label, img_condition=self.img_condition, projection_matrix=self.projection_matrix, kernel_size=self.kernel_size, x_self_cond=x_partial_cond)
        
        if df_type == "x0":
            if self.load_partial and unet_type == "lr":
                mask = (self.split_small_partial == 1)
                masked_denoised = output * mask
                masked_partial = self.split_small_partial * mask
                
                # reduction="mean" calculates also the mean over the locations where there is nothing in the partial split. That is why we do sum and then afterwards normalize by the valid locations
                masked_loss = F.mse_loss(masked_denoised, masked_partial, reduction='sum')
                num_valid_voxels = mask.sum() + 1e-8
                masked_loss = masked_loss / num_valid_voxels

                return F.mse_loss(output, input_data), masked_loss
            else:
                return F.mse_loss(output, input_data)
        elif df_type == "eps":
            # x_start = (noised_data - output * batch_sigma) / batch_alpha.clamp(min=1e-8)
            # self.output = self.autoencoder_module.decode_code(x_start, doctree_in)
            # self.get_sdfs(self.output['neural_mpu'], self.batch_size, bbox = None)
            # self.export_mesh(save_dir = "mytools/octree", index = 0)

            # self.output = self.autoencoder_module.decode_code(input_data, doctree_in)
            # self.get_sdfs(self.output['neural_mpu'], self.batch_size, bbox = None)
            # self.export_mesh(save_dir = "mytools/octree", index = 2)
            if activate_partial_pcd_condition:
                raise ValueError("Not implemented when doing partial conditioning")
            return F.mse_loss(output, noise)
        else:
            raise ValueError(f'invalid loss type {df_type}')
        
    def forward(self):

        self.df.train()

        c = None        

        self.df_hr_loss = torch.tensor(0., device=self.device)
        self.df_lr_loss = torch.tensor(0., device=self.device)

        if self.stage_flag == "lr":
            # self.df_lr_loss = self.forward_lr(split_small)
            batch_id = torch.arange(0, self.batch_size, device=self.device).long()
            if self.load_partial:
                mse_loss, partial_loss = self.calc_loss(self.split_small, None, batch_id, "lr", None, self.df_type[0])
                self.partial_loss = partial_loss

                if self.use_partial_pcd_condition:
                    self.df_lr_loss = mse_loss + self.partial_cond_weight * partial_loss
                else:
                    self.df_lr_loss = mse_loss
            else:
                self.df_lr_loss = self.calc_loss(self.split_small, None, batch_id, "lr", None, self.df_type[0])
        elif self.stage_flag == "hr":
            with torch.no_grad():
                self.input_data, self.doctree_in = self.autoencoder_module.extract_code(self.octree_in)
            # self.df_hr_loss = self.forward_hr(self.input_data, self.small_depth, "hr", self.df_module.unet_lr)
            self.df_hr_loss = self.calc_loss(self.input_data, self.doctree_in, self.doctree_in.batch_id(self.small_depth), "hr", self.df_module.unet_lr, self.df_type[1])

        self.loss = self.df_lr_loss + self.df_hr_loss

    def compute_validation_loss(self):
        self.df.eval()

        df_hr_loss = torch.tensor(0., device=self.device)
        df_lr_loss = torch.tensor(0., device=self.device)

        if self.stage_flag == "lr":
            batch_id = torch.arange(0, self.batch_size, device=self.device).long()

            if self.load_partial:
                mse_loss, partial_loss = self.calc_loss(
                    self.split_small, None, batch_id, "lr", None, self.df_type[0]
                )

                if self.use_partial_pcd_condition:
                    df_lr_loss = mse_loss + self.partial_cond_weight * partial_loss
                else:
                    df_lr_loss = mse_loss
            else:
                df_lr_loss = self.calc_loss(
                    self.split_small, None, batch_id, "lr", None, self.df_type[0]
                )

        elif self.stage_flag == "hr":
            with torch.no_grad():
                input_data, doctree_in = \
                    self.autoencoder_module.extract_code(self.octree_in)

            df_hr_loss = self.calc_loss(
                input_data,
                doctree_in,
                doctree_in.batch_id(self.small_depth),
                "hr",
                self.df_module.unet_lr,
                self.df_type[1]
            )

        return df_lr_loss + df_hr_loss

    def get_sampling_timesteps(self, batch, device, steps):
        times = torch.linspace(1., 0., steps + 1, device=device)
        times = repeat(times, 't -> b t', b=batch)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim=0)
        times = times.unbind(dim=-1)
        return times

    @torch.no_grad()
    def sample_loop(self, doctree_lr = None, ema=False, shape=None, ddim_steps=200, label=None, unet_type="lr", unet_lr=None, df_type="x0", truncated_index=0.0):
        batch_size = self.vq_conf.data.test.batch_size

        time_pairs = self.get_sampling_timesteps(
            batch_size, device=self.device, steps=ddim_steps)

        is_partial_zero_shot_activated       = self.use_partial_pcd_zero_shot       and unet_type == "lr"
        is_partial_zero_shot_clean_activated = self.use_partial_pcd_zero_shot_clean and unet_type == "lr"

        if is_partial_zero_shot_activated or is_partial_zero_shot_clean_activated:
            partial_mask = self.split_small_partial == 1

        if is_partial_zero_shot_activated:
            # TODO work in progress
            # render_mask_8_pyvista(partial_mask, title="Before Fill (Original Mask)")
            # render_mask_only(partial_mask, title="Mask Only")
            # data = compute_visible_connections(partial_mask)
            # render_voxel_scene(
            #     occupancy=data["occupancy"],
            #     mask_points=data["mask_points"],
            #     top_points=data["top_points"],
            #     visible_pairs=None,
            #     connection_voxels=data["connection_voxels"],
            #     title="Mask with Visible Connections"
            # )
            # partial_mask = torch.from_numpy(data["augmented"]).bool().to(partial_mask.device)
            # Extend mask upwards, since we know that there can't be geometry above
            # partial_mask, _ = fill_mask_8_upward(partial_mask)
            # render_mask_8_pyvista(partial_mask, title="After Fill (Upward Filled Mask)")

            t_initial = time_pairs[0][0]  # First element of the first tuple (t, t_next)

            log_snr_initial = self.log_snr(t_initial)

            padded_log_snr = right_pad_dims_to(self.split_small_partial, log_snr_initial)
            alpha_initial, sigma_initial = log_snr_to_alpha_sigma(padded_log_snr)

            # Initialize noised_data as a noisy version of split_small_partial. See blended diffusion
            noised_data = alpha_initial * self.split_small_partial + sigma_initial * torch.randn(shape, device=self.device)
        elif is_partial_zero_shot_clean_activated:
            # ZODIAC-clean: observed voxels start from clean partial, unobserved from pure noise
            noised_data = torch.randn(shape, device=self.device)
            noised_data = partial_mask * self.split_small_partial + ~partial_mask * noised_data
        else:
            noised_data = torch.randn(shape, device=self.device)

        x_start = None

        time_iter = tqdm(time_pairs, desc='small sampling loop time step')

        for t, t_next in time_iter:

            log_snr = self.log_snr(t)
            log_snr_next = self.log_snr(t_next)
            noise_cond = log_snr

            x_cond = x_start
            if self.use_partial_pcd_condition and unet_type == "lr":
                x_cond = self.split_small_partial
            
            if ema:
                output = self.ema_df(unet_type=unet_type, x=noised_data, doctree=doctree_lr,  timesteps=noise_cond, unet_lr=unet_lr, x_self_cond=x_cond, label=label, img_condition=self.img_condition, projection_matrix=self.projection_matrix, kernel_size=self.kernel_size)
            else:
                output = self.df(unet_type=unet_type, x=noised_data, doctree=doctree_lr,  timesteps=noise_cond, unet_lr=unet_lr, x_self_cond=x_cond, label=label, img_condition=self.img_condition, projection_matrix=self.projection_matrix, kernel_size=self.kernel_size)

            if t[0] < truncated_index and unet_type == "lr":
                output.sign_()

            if df_type == "x0":
                x_start = output
                # 'right_pad_dims_to' makes that second argument broadcastable for the first
                padded_log_snr, padded_log_snr_next = map(
                    partial(right_pad_dims_to, noised_data), (log_snr, log_snr_next))

                alpha, sigma = log_snr_to_alpha_sigma(padded_log_snr)
                alpha_next, sigma_next = log_snr_to_alpha_sigma(padded_log_snr_next)

                c = -expm1(padded_log_snr - padded_log_snr_next)
                mean = alpha_next * (noised_data * (1 - c) / alpha + c * output)
                variance = (sigma_next ** 2) * c
                noise = torch.where(
                    # rearrange(t_next > truncated_index, 'b -> b 1 1 1 1'),
                    right_pad_dims_to(noised_data, t_next > truncated_index),
                    torch.randn_like(noised_data),
                    torch.zeros_like(noised_data)
                )
                noised_data = mean + torch.sqrt(variance) * noise
                
                if is_partial_zero_shot_activated:
                    noised_split_partial = alpha_next * self.split_small_partial + sigma_next * torch.randn(shape, device=self.device)
                    noised_data = partial_mask * noised_split_partial + ~partial_mask * noised_data
                elif is_partial_zero_shot_clean_activated:
                    # ZODIAC-clean: pin observed voxels to clean partial at every step
                    noised_data = partial_mask * self.split_small_partial + ~partial_mask * noised_data
            elif df_type == "eps":
                alpha, sigma = log_snr_to_alpha_sigma(log_snr)
                alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)
                alpha, sigma, alpha_next, sigma_next = alpha[0], sigma[0], alpha_next[0], sigma_next[0]
                x_start = (noised_data - output * sigma) / alpha.clamp(min=1e-8)
                noised_data = x_start * alpha_next + output * sigma_next

        return noised_data
    
    @torch.no_grad()
    def sample(self, split_small = None, category = 'airplane', prefix = 'results', ema = False, ddim_steps=200, clean = False, save_index = 0, is_training=True):
        if ema:
            self.ema_df.eval()
        else:
            self.df.eval()
        
        batch_size = self.vq_conf.data.test.batch_size
        if self.enable_label:
            label = torch.ones(batch_size).to(self.device) * category_2_to_label[category]
            label = label.long()
        else:
            label = None
        
        if is_training:
            save_dir = os.path.join(self.opt.logs_dir, self.opt.name, f"{prefix}_training")
        else:
            save_dir = os.path.join(self.opt.logs_dir, self.opt.name, "completion")

        # --- START TIMING ---
        torch.cuda.synchronize()  # make sure GPU is idle before starting timer
        start_time = time.time()

        if split_small == None:
            seed_everything(self.opt.seed + save_index)
            # self.z_shape has shape (8, 16, 16, 16)
            split_small = self.sample_loop(doctree_lr=None, ema=ema, shape=(batch_size, *self.z_shape), ddim_steps=ddim_steps, label=label, unet_type="lr", unet_lr=None, df_type=self.df_type[0], truncated_index=TRUNCATED_TIME)
        
        # self.full_depth = 4
        # self.octree_depth = 6
        # self.input_depth = 8
        octree_small = split2octree_small(split_small, self.octree_depth, self.full_depth)
        # for i in range(batch_size):
        #     save_path = os.path.join(save_dir, "splits_small", f"{save_index}.pth")
        #     os.makedirs(os.path.dirname(save_path), exist_ok=True)
        #     torch.save(split_small[i].unsqueeze(0), save_path)
        
        if self.stage_flag == "lr":
            return
        
        doctree_small = dual_octree.DualOctree(octree_small)
        doctree_small.post_processing_for_docnn()

        doctree_small_num = doctree_small.total_num
        
        seed_everything(self.opt.seed)
        samples = self.sample_loop(doctree_lr=doctree_small, shape=(doctree_small_num, self.code_channel), ema=ema, ddim_steps=ddim_steps, label=label, unet_type="hr", unet_lr=self.ema_df.unet_lr, df_type=self.df_type[1])

        torch.cuda.synchronize()  # wait for all GPU operations to finish
        end_time = time.time()
        total_time_ms = (end_time - start_time) * 1000
        print(f"Total inference time (both phases): {total_time_ms:.1f} ms")

        print(samples.max())
        print(samples.min())
        print(samples.mean())
        print(samples.std())

        # decode z
        self.output = self.autoencoder_module.decode_code(samples, doctree_small)
        self.get_sdfs(self.output['neural_mpu'], batch_size, bbox = None)

        if is_training:
            self.export_mesh(save_dir = save_dir, index = save_index, clean = clean)
            self.export_octree(octree_small, depth = self.small_depth, save_dir = os.path.join(save_dir, "octree"), index = save_index)
        else:
            self.export_test_mesh(save_dir)
            self.export_test_octree(octree_small, depth = self.small_depth, save_dir = os.path.join(save_dir, "lr-octree"), index = save_index)

    def export_octree(self, octree, depth, save_dir = None, index = 0, filename=None):
        try:
            os.makedirs(save_dir, exist_ok=True)
        except FileExistsError:
            pass

        batch_id = octree.batch_id(depth = depth, nempty = False)
        data = torch.ones((len(batch_id), 1), device = self.device)
        data = octree2voxel(data = data, octree = octree, depth = depth, nempty = False)
        data = data.permute(0,4,1,2,3).contiguous()

        batch_size = octree.batch_size

        if filename is None:
            output_name = index
        else:
            output_name = filename

        for i in tqdm(range(batch_size)):
            voxel = data[i].squeeze().cpu().numpy()
            mesh = voxel2mesh(voxel)
            if batch_size == 1:
                mesh.export(os.path.join(save_dir, f'{output_name}.obj'))
            else:
                raise ValueError("TODO: reconsider naming")
                mesh.export(os.path.join(save_dir, f'{index + i}.obj'))

    def export_test_octree(self, octree, depth, save_dir = None, index = 0, filename=None):
        try:
            os.makedirs(save_dir, exist_ok=True)
        except FileExistsError:
            pass

        batch_id = octree.batch_id(depth = depth, nempty = False)
        data = torch.ones((len(batch_id), 1), device = self.device)
        data = octree2voxel(data = data, octree = octree, depth = depth, nempty = False)
        data = data.permute(0,4,1,2,3).contiguous()

        batch_size = octree.batch_size
        mesh_scale=self.vq_conf.data.test.point_scale

        if batch_size != 1:
            raise ValueError("I only can generate mesh and octree for batch size 1!")

        unbatched_filename = self.filename[0]
        # filenames may be subpaths (e.g. a combined filelist spanning datasets) -> ensure the dir exists
        os.makedirs(os.path.join(save_dir, os.path.dirname(unbatched_filename)), exist_ok=True)
        voxel = data[0].squeeze().cpu().numpy()
        mesh = voxel2mesh(voxel)
        vtx = mesh.vertices
        vtx = vtx * self.bbmax # normalize to [self.bbmax;self.bbmin] since the voxel mesh is in [-1;1]
        vtx = vtx * mesh_scale
        mesh = trimesh.Trimesh(vtx, mesh.faces)
        mesh.export(os.path.join(save_dir, f'{unbatched_filename}_voxel.obj'))

        self._export_octree_wireframe(octree, depth, self.bbmax * mesh_scale, os.path.join(save_dir, f'{unbatched_filename}_wireframe.obj'))

    def _export_octree_wireframe(self, octree, depth, factor, file_path):
        dim_max = 2 ** depth
        voxel_size = (2*factor) / dim_max
        h = voxel_size / 2

        cube_offsets = np.array([
            [-h,-h,-h], [h,-h,-h], [h,h,-h], [-h,h,-h],
            [-h,-h,h],  [h,-h,h],  [h,h,h],  [-h,h,h]
        ])
        edges = [(0,1),(1,2),(2,3),(3,0), (4,5),(5,6),(6,7),(7,4), (0,4),(1,5),(2,6),(3,7)]
        
        x, y, z, b = octree.xyzb(depth=0, nempty=False)
        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        z = z.detach().cpu().numpy()

        # normalize to [0,1] using max depth
        x_norm = x / (dim_max - 1)
        y_norm = y / (dim_max - 1)
        z_norm = z / (dim_max - 1)
        
        # scale to [-factor, factor]
        x_scaled = x_norm * 2 * factor - factor
        y_scaled = y_norm * 2 * factor - factor
        z_scaled = z_norm * 2 * factor - factor

        verts = []
        lines = []
        vertex_idx = 1

        for cx, cy, cz in zip(x_scaled, y_scaled, z_scaled):
            r, g, b = 0.0 ,0.0, 1.0
            corners = cube_offsets + np.array([cx, cy, cz])
            for corner in corners:
                verts.append(f"v {corner[0]} {corner[1]} {corner[2]} {r} {g} {b}")
            for i,j in edges:
                lines.append(f"l {vertex_idx+i} {vertex_idx+j}")
            vertex_idx += 8

        with open(file_path, "w") as f:
            f.write("\n".join(verts + lines))
        print(f"Wireframe exported to {file_path}")
    
    

    def get_sdfs(self, neural_mpu, batch_size, bbox):
        # bbox used for marching cubes
        if bbox is not None:
            self.bbmin, self.bbmax = bbox[:3], bbox[3:]
        else:
            sdf_scale = self.solver.sdf_scale
            self.bbmin, self.bbmax = -sdf_scale, sdf_scale    # sdf_scale = 0.8

        self.sdfs = calc_sdf(neural_mpu, batch_size, size = self.solver.resolution, bbmin = self.bbmin, bbmax = self.bbmax)

    def export_mesh(self, save_dir, index = 0, level = 0, clean = False):
        try:
            os.makedirs(save_dir, exist_ok=True)
        except FileExistsError:
            pass
        ngen = self.sdfs.shape[0]
        size = self.solver.resolution
        mesh_scale=self.vq_conf.data.test.point_scale
        for i in range(ngen):
            filename = os.path.join(save_dir, f'{index + i}.obj')
            if ngen == 1:
                filename = os.path.join(save_dir, f'{index}.obj')
            sdf_value = self.sdfs[i].cpu().numpy()
            vtx, faces = np.zeros((0, 3)), np.zeros((0, 3))
            try:
                vtx, faces, _, _ = skimage.measure.marching_cubes(sdf_value, level)
            except:
                pass
            if vtx.size == 0 or faces.size == 0:
                print('Warning from marching cubes: Empty mesh!')
                return
            vtx = vtx * ((self.bbmax - self.bbmin) / size) + self.bbmin   # [0, sz] -> [bbmin, bbmax] Scale the vertex into the range [bbmin, bbmax]
            vtx = vtx * mesh_scale
            mesh = trimesh.Trimesh(vtx, faces)  # Use Trimesh to create a mesh and save it as an OBJ file.
            if clean:
                components = mesh.split(only_watertight=False)
                bbox = []
                for c in components:
                    bbmin = c.vertices.min(0)
                    bbmax = c.vertices.max(0)
                    bbox.append((bbmax - bbmin).max())
                max_component = np.argmax(bbox)
                mesh = components[max_component]
            mesh.export(filename)

    def export_test_mesh(self, save_dir, level = 0, clean = False):
        try:
            os.makedirs(save_dir, exist_ok=True)
        except FileExistsError:
            pass
        ngen = self.sdfs.shape[0]
        size = self.solver.resolution
        mesh_scale=self.vq_conf.data.test.point_scale
        for i in range(ngen):
            if ngen == 1:
                unbatched_filename = self.filename[0]
                filename = os.path.join(save_dir, f'{unbatched_filename}.obj')
                # filenames may be subpaths (e.g. a combined filelist spanning datasets) -> ensure the dir exists
                os.makedirs(os.path.dirname(filename), exist_ok=True)
            else:
                raise ValueError("Not allowed mutliple generations at once due to naming")
            sdf_value = self.sdfs[i].cpu().numpy()

            # Save the SDF grid
            sdf_filename = os.path.join(save_dir, f"{unbatched_filename}_sdf.h5")
            with h5py.File(sdf_filename, 'w') as hf:
                hf.create_dataset('sdf', data=sdf_value, compression='gzip')

            vtx, faces = np.zeros((0, 3)), np.zeros((0, 3))
            try:
                vtx, faces, _, _ = skimage.measure.marching_cubes(sdf_value, level)
            except:
                pass
            if vtx.size == 0 or faces.size == 0:
                print('Warning from marching cubes: Empty mesh!')
                return
            vtx = vtx * ((self.bbmax - self.bbmin) / size) + self.bbmin   # [0, sz] -> [bbmin, bbmax] Scale the vertex into the range [bbmin, bbmax]
            vtx = vtx * mesh_scale

            if self.use_partial_pcd_condition or self.use_partial_pcd_zero_shot:
                partial_points = self.partial_points[0].points
                partial_points = partial_points * mesh_scale
                partial_points_np = partial_points.cpu().numpy()
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(partial_points_np)
                o3d.io.write_point_cloud(os.path.join(save_dir, f"{unbatched_filename}_partial_pcd.ply"), pcd)

            if self.use_sketch_condition:
                image = Image.open(self.edge_path)
                sketch_view = self.edge_path.split("/")[-1]
                image.save(os.path.join(save_dir, f"{unbatched_filename}_{sketch_view}"))

            mesh = trimesh.Trimesh(vtx, faces)  # Use Trimesh to create a mesh and save it as an OBJ file.
            if clean:
                components = mesh.split(only_watertight=False)
                bbox = []
                for c in components:
                    bbmin = c.vertices.min(0)
                    bbmax = c.vertices.max(0)
                    bbox.append((bbmax - bbmin).max())
                max_component = np.argmax(bbox)
                mesh = components[max_component]
            mesh.export(filename)

    def backward(self):

        self.loss.backward()

    def update_EMA(self):
        update_moving_average(self.ema_df, self.df, self.ema_updater)

    def optimize_parameters(self):

        # self.set_requires_grad([self.df.unet_hr], requires_grad=True)

        self.forward()
        assert not torch.isnan(self.loss).any()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()
        self.update_EMA()

    def get_current_errors(self):
        ret = OrderedDict([
            ('loss', self.loss.data),
            ('lr', self.optimizer.param_groups[0]['lr']),
        ])

        if self.stage_flag == "lr" and self.load_partial:
            ret['partial_loss'] = self.partial_loss.data

        if hasattr(self, 'loss_gamma'):
            ret['gamma'] = self.loss_gamma.data

        return ret

    def save(self, label, global_iter):

        state_dict = {
            'df_unet_lr': self.df_module.unet_lr.state_dict(),
            'ema_df_unet_lr': self.ema_df.unet_lr.state_dict(),
            'opt': self.optimizer.state_dict(),
            'global_step': global_iter,
        }
        if self.stage_flag == "hr":
            state_dict['df_unet_hr'] = self.df_module.unet_hr.state_dict()
            state_dict['ema_df_unet_hr'] = self.ema_df.unet_hr.state_dict()

        save_filename = 'df_%s.pth' % (label)
        save_path = os.path.join(self.opt.ckpt_dir, save_filename)

        ckpts = os.listdir(self.opt.ckpt_dir)
        ckpts = [ck for ck in ckpts if ck != 'df_steps-latest.pth' and ck != "df_best.pth"]
        ckpts.sort(key=lambda x: int(x[9:-4]))
        if len(ckpts) > self.opt.ckpt_num:
            for ckpt in ckpts[:-self.opt.ckpt_num]:
                os.remove(os.path.join(self.opt.ckpt_dir, ckpt))

        torch.save(state_dict, save_path)

    def load_ckpt(self, ckpt, df, ema_df, load_options=[]):
        map_fn = lambda storage, loc: storage
        if type(ckpt) == str:
            state_dict = torch.load(ckpt, map_location=map_fn)
        else:
            state_dict = ckpt
        
        if "unet_lr" in load_options and "df_unet_lr" in state_dict:
            df.unet_lr.load_state_dict(state_dict['df_unet_lr'])
            ema_df.unet_lr.load_state_dict(state_dict['ema_df_unet_lr'])
            print(colored('[*] weight successfully load unet_lr from: %s' % ckpt, 'blue'))
        if "unet_hr" in load_options and "df_unet_hr" in state_dict:
            df.unet_hr.load_state_dict(state_dict['df_unet_hr'])
            ema_df.unet_hr.load_state_dict(state_dict['ema_df_unet_hr'])
            print(colored('[*] weight successfully load unet_hr from: %s' % ckpt, 'blue'))

        if "opt" in load_options and "opt" in state_dict:
            self.start_iter = state_dict['global_step']
            print(colored('[*] training start from: %d' % self.start_iter, 'green'))
            self.optimizer.load_state_dict(state_dict['opt'])
            print(colored('[*] optimizer successfully restored from: %s' % ckpt, 'blue'))
