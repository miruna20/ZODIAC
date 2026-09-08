from contextlib import contextmanager, ExitStack
import ocnn
import torch
from torch import nn
import numpy as np
import os
from scipy.spatial.transform import Rotation
import trimesh
from skimage.measure import marching_cubes
import argparse
import random
import pyrr
import matplotlib.pyplot as plt

VIT_FEATURE_CHANNEL = 1280
VIT_PATCH_NUMBER = 256
VIEW_IMAGE_RES = 224
CAMERA_CNT = 2
SKETCH_PER_VIEW = 1
SKETCH_NUMBER = 16

category_5_to_label = {
    'airplane': 0,
    'car': 1,
    'chair': 2,
    'table': 3,
    'rifle': 4,
}

category_2_to_label = {
    'spines': 0,
    'sacrum': 1,
}

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def seed_everything(seed):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_data_class_label(data_class):
	if data_class == "chairs":
		label = "03001627"
	elif data_class == "planes":
		label = "02691156"
	elif data_class == "cars":
		label = "02958343"
	elif data_class == "tables":
		label = "04379243"
	elif data_class == "rifles":
		label = "04090263"
	else:
		raise NotImplementedError
	return label

def get_sample_number_for_metric(data_class, metrics= "fid"):
	assert metrics in ["fid", "cov", "fpd"]
	if metrics == "fid" or metrics == "fpd":
		if data_class == "chairs":
			n_sam = 4744
		elif data_class == "planes":
			n_sam = 2831
		elif data_class == "cars":
			n_sam = 5247
		elif data_class == "tables":
			n_sam = 5956
		elif data_class == "rifles":
			n_sam = 1660
		else:
			raise NotImplementedError
	elif metrics == "cov":
		if data_class == "chairs":
			n_sam = 1356 * 5
		elif data_class == "planes":
			n_sam = 809 * 5
		elif data_class == "cars":
			n_sam = 1500 * 5
		elif data_class == "tables":
			n_sam = 1702 * 5
		elif data_class == "rifles":
			n_sam = 475 * 5
		else:
			raise NotImplementedError
	else:
		raise NotImplementedError    
	return n_sam


def scale_to_unit_sphere(mesh, evaluate_metric = False):
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()

    vertices = mesh.vertices - mesh.bounding_box.centroid
    distances = np.linalg.norm(vertices, axis=1)
    vertices /= np.max(distances)
    if evaluate_metric:
        vertices /= 2
    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)


def shapenet_v2_to_v1_orientation(mesh):
	mesh.apply_transform(get_rotation_matrix(-90, 'y'))
	# mesh.invert()
	return mesh

def get_grid_normal(grid, bouding_box_length=2):

    grid_res = grid.shape[-1]
    n = grid_res - 1
    voxel_size = bouding_box_length/n

    X_1 = torch.cat((grid[:, :, 1:, :, :], (3 * grid[:, :, n, :, :] - 3 *
                    grid[:, :, n-1, :, :] + grid[:, :, n-2, :, :]).unsqueeze_(2)), 2)
    X_2 = torch.cat(((-3 * grid[:, :, 1, :, :] + 3 * grid[:, :, 0, :, :] +
                    grid[:, :, 2, :, :]).unsqueeze_(2), grid[:, :, :n, :, :]), 2)
    grid_normal_x = (X_1 - X_2) / (2 * voxel_size)

    Y_1 = torch.cat((grid[:, :, :, 1:, :], (3 * grid[:, :, :, n, :] - 3 *
                    grid[:, :, :, n-1, :] + grid[:, :, :, n-2, :]).unsqueeze_(3)), 3)
    Y_2 = torch.cat(((-3 * grid[:, :, :, 1, :] + 3 * grid[:, :, :, 0, :] +
                    grid[:, :, :, 2, :]).unsqueeze_(3), grid[:, :, :, :n, :]), 3)
    grid_normal_y = (Y_1 - Y_2) / (2 * voxel_size)
    
    Z_1 = torch.cat((grid[:, :, :, :, 1:], (3 * grid[:, :, :, :, n] - 3 *
                    grid[:, :, :, :, n-1] + grid[:, :, :, :, n-2]).unsqueeze_(4)), 4)
    Z_2 = torch.cat(((-3 * grid[:, :, :, :, 1] + 3 * grid[:, :, :, :, 0] +
                    grid[:, :, :, :, 2]).unsqueeze_(4), grid[:, :, :, :, :n]), 4)
    grid_normal_z = (Z_1 - Z_2) / (2 * voxel_size)

    return torch.cat((grid_normal_x, grid_normal_y, grid_normal_z), 1)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def cast_tuple(val, repeat=1):
    return val if isinstance(val, tuple) else ((val,) * repeat)


def run(cmd,verbose=True):
    if verbose:
        print(cmd)
    os.system(cmd)


def process_mesh(mesh, sample_number=2048):
    mesh = scale_to_unit_sphere(mesh=mesh, evaluate_metric=True)
    return mesh.sample(sample_number)


def points_gradient(inputs, outputs):
    d_points = torch.ones_like(
        outputs, requires_grad=False, device=outputs.device)
    points_grad = torch.autograd.grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]
    return points_grad


def get_voxel_coordinates(resolution=32, size=1, center=0, device=None):
    if type(center) == int:
        center = (center, center, center)
    points = np.meshgrid(
		np.linspace(center[0] - size, center[0] + size, resolution),
		np.linspace(center[1] - size, center[1] + size, resolution),
		np.linspace(center[2] - size, center[2] + size, resolution)
    )
    points = np.stack(points)
    points = np.swapaxes(points, 1, 2)
    points = points.reshape(3, -1).transpose()
    if device is not None:
        return torch.tensor(points, dtype=torch.float32, device=device)
    else:
        return torch.tensor(points, dtype=torch.float32)


def process_sdf(volume, level=0, padding=True, spacing=None, offset=-1,normalize=False):
    try:
        if padding:
            volume = np.pad(volume, 1, mode='constant', constant_values=1)
        if spacing is None:
            spacing = 2/(volume.shape[-1] - 1)
        vertices, faces, normals, _ = marching_cubes(
                volume, level=level, spacing=(spacing, spacing, spacing))
        if offset is not None:
            vertices += offset
        if normalize:
            return scale_to_unit_sphere(trimesh.Trimesh(
                vertices=vertices, faces=faces, vertex_normals=normals))     
        else:
            return trimesh.Trimesh(
                vertices=vertices, faces=faces, vertex_normals=normals)
    except Exception as e:
        print(str(e))
        return None


def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


class NanException(Exception):
    pass


def get_rotation_matrix(angle, axis='y'):
	rotation = Rotation.from_euler(axis, angle, degrees=True)
	matrix = np.identity(4)
	matrix[:3, :3] = rotation.as_matrix()
	return matrix

def get_pc_rotation_matrix(angle, axis='y'):
	rotation = Rotation.from_euler(axis, angle, degrees=True)
	return rotation.as_matrix()
	# return matrix
        

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def cast_list(el):
    return el if isinstance(el, list) else [el]


def exists(val):
    return val is not None


@contextmanager
def null_context():
    yield


def combine_contexts(contexts):
    @contextmanager
    def multi_contexts():
        with ExitStack() as stack:
            yield [stack.enter_context(ctx()) for ctx in contexts]

    return multi_contexts


def default(value, d):
    return value if exists(value) else d


def cycle(iterable):
    while True:
        for i in iterable:
            yield i


def cast_list(el):
    return el if isinstance(el, list) else [el]


def is_empty(t):
    if isinstance(t, torch.Tensor):
        return t.nelement() == 0
    return not exists(t)


def raise_if_nan(t):
    if torch.isnan(t):
        raise NanException


def noise(batch_size, latent_dim, device):
    return torch.randn(batch_size, latent_dim).cuda(device)


def noise_list(batch_size, layers, latent_dim, device):
    return [(noise(batch_size, latent_dim, device), layers)]


def mixed_list(batch_size, layers, latent_dim, device):
    tt = int(torch.rand(()).numpy() * layers)
    return noise_list(batch_size, tt, latent_dim, device) + noise_list(batch_size, layers - tt, latent_dim, device)


def latent_to_w(style_vectorizer, latent_descr):
    return [(style_vectorizer(z), num_layers) for z, num_layers in latent_descr]


def image_noise(n, im_size, device):
    return torch.FloatTensor(n, im_size, im_size, 1).uniform_(0., 1.).cuda(device)


def volume_noise(n, vol_size, device):
    return torch.FloatTensor(n, vol_size, vol_size, vol_size, 1).uniform_(0., 1.).cuda(device)


def leaky_relu(p=0.2,):
    return nn.LeakyReLU(p, inplace=True)


def evaluate_in_chunks(max_batch_size, model, *args):
    split_args = list(
            zip(*list(map(lambda x: x.split(max_batch_size, dim=0), args))))
    chunked_outputs = [model(*i) for i in split_args]
    if len(chunked_outputs) == 1:
        return chunked_outputs[0]
    return torch.cat(chunked_outputs, dim=0)


def styles_def_to_tensor(styles_def):
    return torch.cat([t[:, None, :].expand(-1, n, -1) for t, n in styles_def], dim=1)


def set_requires_grad(model, bool):
    for p in model.parameters():
        p.requires_grad = bool

def linear_slerp(val, low, high):
    val = val.squeeze()
    return (1-val)*low + val * high


class TorchRecoder:
    def __init__(self):
        self.total_time = 0
        self.calls = 0
        self.total_memory = 0

    def __enter__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        self.start.record()

    def __exit__(self, *args):
        self.end.record()
        torch.cuda.synchronize()
        self.total_time += self.start.elapsed_time(self.end)
        self.calls += 1
        peak_memory = torch.cuda.memory.max_memory_allocated()    / (2 ** 30)
        self.total_memory += peak_memory

    def reset(self):
        self.total_time = 0
        self.calls = 0
        self.total_memory = 0

    def avg_time(self):
        return self.total_time / self.calls if self.calls > 0 else 0
    
    def avg_memory(self):
        return self.total_memory / self.calls if self.calls > 0 else 0
    
def convert_partial_pcd_2_mesh(input, model, index):
    def points2octree(points, model):
        octree = ocnn.octree.Octree(depth = model.input_depth, full_depth = model.full_depth)
        octree.build_octree(points)
        return octree

    # Convert partial pcd to octree
    points = [pts.cuda(non_blocking=True) for pts in input['partial_points']]
    octrees = [points2octree(pts, model) for pts in points]
    octree = ocnn.octree.merge_octrees(octrees)
    octree.construct_all_neigh()
    # input['split_small_partial'] = octree2split_small(input['octree_in_partial'], model.full_depth)

    # Convert octree to mesh
    save_dir = os.path.join(model.opt.logs_dir, model.opt.name, f"results_partial")
    model.export_octree(octree, model.small_depth, save_dir=os.path.join(save_dir, "octree2voxel"), index=index, filename=input['filename'])

def get_voxel_coordinates(resolution=32, size=1, center=0, padding=True, homogeneous=True):
    if type(center) == int:
        center = (center, center, center)
    points = np.meshgrid(
        np.linspace(center[0] - size, center[0] + size, resolution),
        np.linspace(center[1] - size, center[1] + size, resolution),
        np.linspace(center[2] - size, center[2] + size, resolution)
    )
    points = np.stack(points)
    points = np.swapaxes(points, 1, 2)
    points = points.reshape(3, -1).transpose()
    if padding:
        points = points * (resolution - 1)/resolution
    if homogeneous:
        points = np.concatenate([points, np.ones((points.shape[0], 1))], 1)
    return points

def create_random_pose(rotation=None, elevation=None, return_angles=False):
    if rotation is None:
        rotation = np.random.rand() * 360
    else:
        rotation = float(rotation)

    if elevation is None:
        elevation = np.random.rand() * 40
    else:
        elevation = float(elevation)
    eye = np.array([np.sin(rotation/180*np.pi)*np.cos(elevation/180*np.pi),
                    np.sin(elevation/180*np.pi),
                    np.cos(rotation/180*np.pi)*np.cos(elevation/180*np.pi)]) * 2.5

    target = np.zeros(3)
    camera_pose = np.array(pyrr.Matrix44.look_at(eye=eye,
                                                 target=target,
                                                 up=np.array([0.0, 1.0, 0])).T)
    if return_angles:
        return np.linalg.inv(camera_pose), rotation, elevation
    else:
        return np.linalg.inv(camera_pose)
    
def get_P_from_transform_matrix(matrix):

    location = matrix[0:3, 3]

    rotation = np.transpose(matrix[0:3, 0:3])
    t = np.tan(np.pi / 6.0)
    width = 224

    return np.array([[112 / t, 0, width/2], [0, 112 / t, width/2], [0, 0, 1]]) \
        @ np.concatenate([rotation, np.expand_dims(-1*rotation @ location, 1)], 1)

if CAMERA_CNT == 5:
    Projection_List = [
        get_P_from_transform_matrix(create_random_pose(rotation=90, elevation=20)),
        get_P_from_transform_matrix(create_random_pose(rotation=135, elevation=20)),
        get_P_from_transform_matrix(create_random_pose(rotation=180, elevation=20)),
        get_P_from_transform_matrix(
            create_random_pose(rotation=225, elevation=20)),
        get_P_from_transform_matrix(
            create_random_pose(rotation=270, elevation=20)),
    ]
    Projection_List_zero = [
        get_P_from_transform_matrix(create_random_pose(rotation=90, elevation=0)),
        get_P_from_transform_matrix(create_random_pose(rotation=135, elevation=0)),
        get_P_from_transform_matrix(create_random_pose(rotation=180, elevation=0)),
        get_P_from_transform_matrix(create_random_pose(rotation=225, elevation=0)),
        get_P_from_transform_matrix(create_random_pose(rotation=270, elevation=0)),
    ]
elif CAMERA_CNT == 2:
    Projection_List = [
        get_P_from_transform_matrix(create_random_pose(rotation=90, elevation=20)),
        get_P_from_transform_matrix(create_random_pose(rotation=-90, elevation=20)),
    ]
    Projection_List_zero = [
        get_P_from_transform_matrix(create_random_pose(rotation=90, elevation=0)),
        get_P_from_transform_matrix(create_random_pose(rotation=-90, elevation=0)),
    ]
else:
    raise ValueError

white_image_feature = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "white_image_feature.npy"))

### Space carving

# -------------------------
# UTILITY FUNCTIONS
# -------------------------

def is_visible(start, end, occupancy):
    start = np.array(start, dtype=float)
    end = np.array(end, dtype=float)
    diff = end - start
    steps = int(np.linalg.norm(diff) * 2)

    if steps <= 1:
        return True

    for t in np.linspace(0, 1, steps, endpoint=False)[1:]:
        point = start + t * diff
        ix, iy, iz = np.round(point).astype(int)
        X, Y, Z = occupancy.shape

        if 0 <= ix < X and 0 <= iy < Y and 0 <= iz < Z:
            if occupancy[ix, iy, iz]:
                if not (ix == int(end[0]) and iy == int(end[1]) and iz == int(end[2])):
                    return False
    return True


def get_line_voxels(start, end, shape):
    start = np.array(start, dtype=float)
    end = np.array(end, dtype=float)
    diff = end - start
    steps = int(np.linalg.norm(diff) * 2)

    voxels = []
    if steps <= 1:
        return [tuple(np.round(start).astype(int))]

    for t in np.linspace(0, 1, steps):
        point = start + t * diff
        ix, iy, iz = np.round(point).astype(int)
        X, Y, Z = shape

        if 0 <= ix < X and 0 <= iy < Y and 0 <= iz < Z:
            voxels.append((ix, iy, iz))

    return list(set(voxels))


# -------------------------
# CORE LOGIC
# -------------------------

def compute_visible_connections(mask_8):
    if isinstance(mask_8, torch.Tensor):
        mask_8 = mask_8.detach().cpu().numpy()

    if mask_8.ndim == 5:
        mask_8 = mask_8[0]

    assert mask_8.ndim == 4, f"Expected (8,X,Y,Z), got {mask_8.shape}"

    occupancy = (mask_8 > 0).any(axis=0)
    augmented_occupancy = occupancy.copy()

    mask_points = np.argwhere(occupancy).astype(np.float32)

    X, Y, Z = occupancy.shape
    x_center = (X - 1) / 2
    top_line_points = np.array(
        [[x_center, Y - 1, z] for z in range(Z)],
        dtype=np.float32
    )

    connection_voxels = set()
    visible_pairs = []

    for top_pt in top_line_points:
        for mask_pt in mask_points:
            if is_visible(top_pt, mask_pt, occupancy):
                visible_pairs.append((top_pt, mask_pt))
                voxels_on_line = get_line_voxels(
                    top_pt, mask_pt, occupancy.shape
                )
                connection_voxels.update(voxels_on_line)

    for v in connection_voxels:
        augmented_occupancy[v] = True

    return {
        "occupancy": occupancy,
        "augmented": augmented_occupancy,
        "mask_points": mask_points,
        "top_points": top_line_points,
        "visible_pairs": visible_pairs,
        "connection_voxels": connection_voxels,
    }


# -------------------------
# RENDERING FUNCTIONS
# -------------------------

def render_voxel_scene(
    occupancy,
    mask_points=None,
    top_points=None,
    visible_pairs=None,
    connection_voxels=None,
    title="Voxel Scene"
):
    import pyvista as pv

    plotter = pv.Plotter(notebook=False, window_size=[800, 600])

    X, Y, Z = occupancy.shape
    bounds = [-0.5, X - 0.5,
              -0.5, Y - 0.5,
              -0.5, Z - 0.5]

    # Original mask
    if mask_points is not None:
        plotter.add_points(
            pv.PolyData(mask_points),
            color="green",
            render_points_as_spheres=True,
            point_size=6
        )

    # Bounding box
    plotter.add_mesh(
        pv.Cube(bounds=bounds),
        color='white',
        style='wireframe',
        line_width=2
    )

    # Top points
    if top_points is not None:
        plotter.add_points(
            pv.PolyData(top_points),
            color="red",
            render_points_as_spheres=True,
            point_size=8
        )

    # Visible lines
    if visible_pairs is not None:
        for top_pt, mask_pt in visible_pairs:
            plotter.add_mesh(
                pv.Line(top_pt, mask_pt),
                color="blue",
                opacity=0.3,
                line_width=1
            )

    # Connection voxels
    if connection_voxels:
        new_voxels = np.array(list(connection_voxels)).astype(np.float32)
        plotter.add_points(
            pv.PolyData(new_voxels),
            color="yellow",
            render_points_as_spheres=True,
            point_size=5
        )

    plotter.add_axes()
    plotter.add_text(title, font_size=12)
    plotter.show()

def render_mask_only(mask_8, title="Mask Only"):
    if isinstance(mask_8, torch.Tensor):
        mask_8 = mask_8.detach().cpu().numpy()

    if mask_8.ndim == 5:
        mask_8 = mask_8[0]

    occupancy = (mask_8 > 0).any(axis=0)
    mask_points = np.argwhere(occupancy).astype(np.float32)

    render_voxel_scene(
        occupancy=occupancy,
        mask_points=mask_points,
        title=title
    )

# -------------------------
# MAIN
# -------------------------

if __name__ == "__main__":

    mask_8 = -np.ones((8, 16, 16, 16), dtype=np.int8)

    # Plane at z=11
    mask_8[:, :, :, 11] = 1

    # Smaller 6x6 patch above it at z=13
    patch_size = 6
    x_center = mask_8.shape[1] // 2
    y_center = mask_8.shape[2] // 2

    x_start = x_center - patch_size // 2
    x_end = x_start + patch_size
    y_start = y_center - patch_size // 2
    y_end = y_start + patch_size

    mask_8[:, x_start:x_end, y_start:y_end, 13] = 1

    # Convert to boolean mask
    mask_8_bool = (mask_8 == 1)

    # -----------------
    # Just render mask
    # -----------------
    render_mask_only(mask_8_bool, title="Mask Only")

    # -----------------
    # Compute + render connections
    # -----------------
    data = compute_visible_connections(mask_8_bool)

    render_voxel_scene(
        occupancy=data["occupancy"],
        mask_points=data["mask_points"],
        top_points=data["top_points"],
        visible_pairs=data["visible_pairs"],
        connection_voxels=data["connection_voxels"],
        title="Mask with Visible Connections"
    )

    print("Original occupied voxels:", np.sum(mask_8_bool.any(axis=0)))
    print("After adding connection voxels:", np.sum(data["augmented"]))
