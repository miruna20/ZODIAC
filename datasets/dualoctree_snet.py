# --------------------------------------------------------
# Dual Octree Graph Networks
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import os
import ocnn
import torch
import numpy as np
import copy
from random import random

from ocnn.octree import Octree, Points
from solver import ShapeNetDataset, SpinesDataset
from .utils import collate_func
from utils.util import CAMERA_CNT, SKETCH_PER_VIEW, SKETCH_NUMBER, create_random_pose, get_P_from_transform_matrix, Projection_List_zero, Projection_List


class TransformShape:

    def __init__(self, flags, extra_flags=None):
        self.flags = flags
        self.extra_flags = extra_flags

        self.depth = flags.depth
        self.full_depth = flags.full_depth

        self.point_sample_num = flags.point_sample_num
        self.point_scale = flags.point_scale
        self.noise_std = 0.005

    def points2octree(self, points: Points):
        octree = Octree(self.depth, self.full_depth)
        octree.build_octree(points)
        return octree

    def process_points_cloud(self, sample):
        # get the input
        points, normals = sample['points'], sample['normals']
        points = points / self.point_scale    # scale to [-1.0, 1.0], because in the preprocessing file, we save the positions of the sampled SDF values in [-0.5, 0.5] range

        # transform points to octree
        points_gt = Points(points = torch.from_numpy(points).float(), normals = torch.from_numpy(normals).float())
        if self.flags.load_color:
            points_gt.features = torch.from_numpy(sample['colors']).float()
        points_gt.clip(min=-1, max=1)

        return {'points': points_gt}
    
    def process_partial_points_cloud(self, sample):
        # get the input
        points, normals = sample['points'], sample['normals']
        points = points / self.point_scale    # scale to [-mesh_scale, mesh_scale], because in the preprocessing file, we save the positions of the sampled SDF values in [-mesh_scale*shape_scale, mesh_scale*shape_scale] range

        # transform points to octree
        points_gt = Points(points = torch.from_numpy(points).float(), normals = torch.from_numpy(normals).float())
        if self.flags.load_color:
            points_gt.features = torch.from_numpy(sample['colors']).float()
        points_gt.clip(min=-1, max=1)

        return {'partial_points': points_gt}

    def sample_sdf(self, sample):     # The coordinates of the loaded SDF here are also within the range [-1, 1].
        sdf = sample['sdf']
        grad = sample['grad']
        points = sample['points'] / self.point_scale    # to [-1, 1]

        rand_idx = np.random.choice(points.shape[0], size=self.point_sample_num)
        points = torch.from_numpy(points[rand_idx]).float()
        sdf = torch.from_numpy(sdf[rand_idx]).float()
        grad = torch.from_numpy(grad[rand_idx]).float()
        return {'pos': points, 'sdf': sdf, 'grad': grad}

    def sample_on_surface(self, points, normals):
        rand_idx = np.random.choice(points.shape[0], size=self.point_sample_num)
        xyz = torch.from_numpy(points[rand_idx]).float()
        grad = torch.from_numpy(normals[rand_idx]).float()
        sdf = torch.zeros(self.point_sample_num)
        return {'pos': xyz, 'sdf': sdf, 'grad': grad}

    def sample_off_surface(self, xyz):
        xyz = xyz / self.point_scale    # to [-1, 1]

        rand_idx = np.random.choice(xyz.shape[0], size=self.point_sample_num)
        xyz = torch.from_numpy(xyz[rand_idx]).float()
        # grad = torch.zeros(self.sample_number, 3)    # dummy grads
        grad = xyz / (xyz.norm(p=2, dim=1, keepdim=True) + 1.0e-6)
        sdf = -1 * torch.ones(self.point_sample_num)    # dummy sdfs
        return {'pos': xyz, 'sdf': sdf, 'grad': grad}

    def __call__(self, sample, idx):
        output = {}

        if self.flags.load_octree:
            output['octree_in'] = sample['octree_in']

        if self.flags.load_pointcloud:
            output = self.process_points_cloud(sample['point_cloud'])

        if self.flags.load_partial:
            partial = self.process_partial_points_cloud(sample['partial'])
            output.update(partial)

        if self.flags.load_split_small:
            output['split_small'] = sample['split_small']

        if self.flags.load_split_large:
            output['split_large'] = sample['split_large']

        # sample ground truth sdfs
        if self.flags.load_sdf:
            sdf_samples = self.sample_sdf(sample['sdf'])
            output.update(sdf_samples)

        # sample on surface points and off surface points
        if self.flags.sample_surf_points:
            on_surf = self.sample_on_surface(sample['points'], sample['normals'])
            off_surf = self.sample_off_surface(sample['sdf']['points'])    # TODO
            sdf_samples = {
                    'pos': torch.cat([on_surf['pos'], off_surf['pos']], dim=0),
                    'grad': torch.cat([on_surf['grad'], off_surf['grad']], dim=0),
                    'sdf': torch.cat([on_surf['sdf'], off_surf['sdf']], dim=0)}
            output.update(sdf_samples)

        if self.extra_flags.get("use_sketch_condition", False):
            output["img_condition"] = torch.from_numpy(sample["image_feature"]).float()
            output["projection_matrix"] = torch.from_numpy(sample["projection_matrix"]).float()
            output["feature_folder"] = sample["feature_folder"]
            output["edge_path"] = sample["edge_path"]
        return output


class ReadFile:
    def __init__(self, flags, extra_flags=None):
        self.load_octree = flags.load_octree
        self.load_pointcloud = flags.load_pointcloud
        self.load_partial = flags.load_partial
        self.load_split_small = flags.load_split_small
        self.load_split_large = flags.load_split_large
        self.load_occu = flags.load_occu
        self.load_sdf = flags.load_sdf
        self.load_color = flags.load_color

        # Sketch conditioning (unused in ZODIAC; kept inert for OctFusion compatibility).
        # sketch_folder was removed from the configs, so read it only when actually enabled.
        self.use_sketch_condition = extra_flags.get("use_sketch_condition", False)
        self.detail_view = extra_flags.detail_view
        self.elevation_zero = extra_flags.elevation_zero
        self.feature_drop_out = extra_flags.feature_drop_out
        self.feature_folder = None
        self.edge_path = None

        if self.use_sketch_condition:
            sketch_folder = extra_flags.sketch_folder
            self.feature_folder = os.path.join(sketch_folder, "feature")
            self.edge_path = os.path.join(sketch_folder, "edge")
            self.white_image_feature = np.load(os.path.join(self.feature_folder, "white_image_feature.npy"))
            if not self.detail_view:
                if self.elevation_zero:
                    self.projection_list = Projection_List_zero
                else:
                    self.projection_list = Projection_List
            else:
                self.pm_folder = os.path.join(sketch_folder, "projection_matrix")

    def __call__(self, filename):
        output = {}

        if self.load_octree:
            octree_path = os.path.join(filename, 'octree.pth')
            raw = torch.load(octree_path)
            octree_in = raw['octree_in']
            output['octree_in'] = octree_in

        if self.load_pointcloud:
            filename_pc = os.path.join(filename, 'pointcloud.npz')
            raw = np.load(filename_pc)
            point_cloud = {'points': raw['points'], 'normals': raw['normals']}
            if self.load_color:
                filename_color = os.path.join(filename, 'color.npz')
                raw = np.load(filename_color)
                point_cloud['colors'] = raw['colors']
            else:
                point_cloud['colors'] = None
            output['point_cloud'] = point_cloud

        if self.load_partial:
            filename_partial = os.path.join(filename, 'partial.npz')
            raw = np.load(filename_partial)
            partial = {'points': raw['points'], 'normals': raw['normals']}
            output['partial'] = partial

        if self.load_split_small:
            filename_split_small = os.path.join(filename, 'split_small.pth')
            raw = torch.load(filename_split_small, map_location = 'cpu')
            output['split_small'] = raw

        if self.load_split_large:
            filename_split_large = os.path.join(filename, 'split_large.pth')
            try:
                raw = torch.load(filename_split_large, map_location = 'cpu')
            except:
                print('Error!!')
                print(filename)
            output['split_large'] = raw

        if self.load_occu:
            filename_occu = os.path.join(filename, 'points.npz')
            raw = np.load(filename_occu)
            occu = {'points': raw['points'], 'occupancies': raw['occupancies']}
            output['occu'] = occu

        if self.load_sdf:
            filename_sdf = os.path.join(filename, 'sdf.npz')
            raw = np.load(filename_sdf)
            sdf = {'points': raw['points'], 'grad': raw['grad'], 'sdf': raw['sdf']}
            output['sdf'] = sdf

        if self.use_sketch_condition:
            model_name = filename.split("/")[-1]
            if not self.detail_view:
                sketch_view_index = np.random.randint(0, CAMERA_CNT * SKETCH_PER_VIEW)
                try:
                    image_feature = np.load(os.path.join(
                        self.feature_folder, model_name, f"{sketch_view_index:02d}.npy"))
                except Exception as e:
                    print(str(e))
                    image_feature = self.white_image_feature
                pm = self.projection_list[sketch_view_index//SKETCH_PER_VIEW]
            else:
                sketch_view_index = np.random.randint(0, SKETCH_NUMBER)
                try:
                    image_feature = np.load(os.path.join(self.feature_folder, model_name))[
                        sketch_view_index]
                    pm = np.load(os.path.join(self.pm_folder, model_name))[
                        sketch_view_index]
                except Exception as e:
                    print(str(e))
                    image_feature = self.white_image_feature
                    pm = get_P_from_transform_matrix(create_random_pose())
                    projection_matrix = np.expand_dims(pm, 0)

            if random() < self.feature_drop_out:
                image_feature = self.white_image_feature

            self.vit_global = False # default value in LAS-Diffusion
            self.vit_local = True # default value in LAS-Diffusion

            if self.vit_global and not self.vit_local:
                image_feature = image_feature[0:1]
            elif self.vit_local and not self.vit_global:
                image_feature = image_feature[1:]

            output['image_feature'] = image_feature

            projection_matrix = np.expand_dims(pm, 0)
            output['projection_matrix'] = projection_matrix
            output['feature_folder'] = os.path.join(self.feature_folder, model_name)
            output['edge_path'] = os.path.join(self.edge_path, model_name)
        return output


def get_shapenet_dataset(flags):
    transform = TransformShape(flags)
    read_file = ReadFile(flags)
    dataset = ShapeNetDataset(flags.location, flags.filelist, transform,
                                        read_file=read_file, in_memory=flags.in_memory)
    return dataset, collate_func

def get_spines_dataset(flags, extra_flags):
    transform = TransformShape(flags, extra_flags)
    read_file = ReadFile(flags, extra_flags)
    dataset = SpinesDataset(flags.location, flags.filelist, transform, read_file=read_file, in_memory=flags.in_memory)
    return dataset, collate_func
