import torch
import torch.nn.functional as F
from genrec.utils.typing import *

def get_ray_directions(
    H: int,
    W: int,
    focal: Union[float, Tuple[float, float]],
    principal: Optional[Tuple[float, float]] = None,
    use_pixel_centers: bool = True,
) -> Float[Tensor, "H W 3"]:
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        H: image height;
        W: image width;
        focal: focal length;
        principal: principal point;
        use_pixel_centers: whether use pixel centers
    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate, 
                    where the x-axis rannge from [-1, 1],
                    y-axis range from [-1, 1], and
                    z-axis points outside of the camera.
    """
    pixel_center = 0.5 if use_pixel_centers else 0

    if isinstance(focal, float):
        fx, fy = focal, focal
        cx, cy = W / 2, H / 2
    else:
        fx, fy = focal
        assert principal is not None
        cx, cy = principal

    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32) + pixel_center,
        torch.arange(H, dtype=torch.float32) + pixel_center,
        indexing="xy",
    )

    directions: Float[Tensor, "H W 3"] = torch.stack(
        [(i - cx) / fx, (j - cy) / fy, torch.ones_like(i)], dim=-1
    )

    return directions

def get_rays(
    directions: Float[Tensor, "... 3"],
    c2w: Float[Tensor, "... 4 4"],
    keepdim=False,
    noise_scale=0.0,
) -> Tuple[Float[Tensor, "... 3"], Float[Tensor, "... 3"]]:
    # Rotate ray directions from camera coordinate to the world coordinate
    assert directions.shape[-1] == 3

    if directions.ndim == 2:  # (N_rays, 3)
        if c2w.ndim == 2:  # (4, 4)
            c2w = c2w[None, :, :]
        assert c2w.ndim == 3  # (N_rays, 4, 4) or (1, 4, 4)
        rays_d = (directions[:, None, :] * c2w[:, :3, :3]).sum(-1)  # (N_rays, 3)
        rays_o = c2w[:, :3, 3].expand(rays_d.shape)
    elif directions.ndim == 3:  # (H, W, 3)
        assert c2w.ndim in [2, 3]
        if c2w.ndim == 2:  # (4, 4)
            rays_d = (directions[:, :, None, :] * c2w[None, None, :3, :3]).sum(
                -1
            )  # (H, W, 3)
            rays_o = c2w[None, None, :3, 3].expand(rays_d.shape)
        elif c2w.ndim == 3:  # (B, 4, 4)
            rays_d = (directions[None, :, :, None, :] * c2w[:, None, None, :3, :3]).sum(
                -1
            )  # (B, H, W, 3)
            rays_o = c2w[:, None, None, :3, 3].expand(rays_d.shape)
    elif directions.ndim == 4:  # (B, H, W, 3)
        assert c2w.ndim == 3  # (B, 4, 4)
        rays_d = (directions[:, :, :, None, :] * c2w[:, None, None, :3, :3]).sum(
            -1
        )  # (B, H, W, 3)
        rays_o = c2w[:, None, None, :3, 3].expand(rays_d.shape)

    # add camera noise to avoid grid-like artifect
    # https://github.com/ashawkey/stable-dreamfusion/blob/49c3d4fa01d68a4f027755acf94e1ff6020458cc/nerf/utils.py#L373
    if noise_scale > 0:
        rays_o = rays_o + torch.randn(3, device=rays_o.device) * noise_scale
        rays_d = rays_d + torch.randn(3, device=rays_d.device) * noise_scale

    rays_d = F.normalize(rays_d, dim=-1)
    if not keepdim:
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

    return rays_o, rays_d

def get_plucker_rays(
    directions: Float[Tensor, "... 3"],
    c2w: Float[Tensor, "... 4 4"],
    keepdim=False,
    noise_scale=0.0,
) -> Float[Tensor, "... 6"]:
    # Rotate ray directions from camera coordinate to the world coordinate
    assert directions.shape[-1] == 3

    if directions.ndim == 2:  # (N_rays, 3)
        if c2w.ndim == 2:  # (4, 4)
            c2w = c2w[None, :, :]
        assert c2w.ndim == 3  # (N_rays, 4, 4) or (1, 4, 4)
        rays_d = (directions[:, None, :] * c2w[:, :3, :3]).sum(-1)  # (N_rays, 3)
        rays_o = c2w[:, :3, 3].expand(rays_d.shape)
    elif directions.ndim == 3:  # (H, W, 3)
        assert c2w.ndim in [2, 3]
        if c2w.ndim == 2:  # (4, 4)
            rays_d = (directions[:, :, None, :] * c2w[None, None, :3, :3]).sum(
                -1
            )  # (H, W, 3)
            rays_o = c2w[None, None, :3, 3].expand(rays_d.shape)
        elif c2w.ndim == 3:  # (B, 4, 4)
            rays_d = (directions[None, :, :, None, :] * c2w[:, None, None, :3, :3]).sum(
                -1
            )  # (B, H, W, 3)
            rays_o = c2w[:, None, None, :3, 3].expand(rays_d.shape)
    elif directions.ndim == 4:  # (B, H, W, 3)
        assert c2w.ndim == 3  # (B, 4, 4)
        rays_d = (directions[:, :, :, None, :] * c2w[:, None, None, :3, :3]).sum(
            -1
        )  # (B, H, W, 3)
        rays_o = c2w[:, None, None, :3, 3].expand(rays_d.shape)

    # add camera noise to avoid grid-like artifect
    # https://github.com/ashawkey/stable-dreamfusion/blob/49c3d4fa01d68a4f027755acf94e1ff6020458cc/nerf/utils.py#L373
    if noise_scale > 0:
        rays_o = rays_o + torch.randn(3, device=rays_o.device) * noise_scale
        rays_d = rays_d + torch.randn(3, device=rays_d.device) * noise_scale

    rays_d = F.normalize(rays_d, dim=-1)
    if not keepdim:
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

    rays_dxo = torch.cross(rays_o, rays_d, dim=-1)                  # B, H, W, 3
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)                 # B, H, W, 6
    return plucker

def bbox_get_rays(H: int, W: int, K: Float[Tensor, "3 3"], c2w: Float[Tensor, "4 4"], normalize_dir: bool=False, format: str = "OpenCV"):
    """ calculate rays for bounding box rasterization
    The differences between this and get_rays are:
        1. the i and j are transposed
        2. the rays are not normalized
    Args:
        H (int): image height
        W (int): image width
        K (Float[Tensor, &quot;3 3&quot;]): caermra intrinsics
        c2w (Float[Tensor, &quot;4 4&quot;]): caemra extrinsics
        normalize_dir (bool, optional): _description_. Defaults to False.
        format (str, optional): _description_. Defaults to "OpenCV".

    Returns:
        ray origin: (H, W, 3)
        ray direction: (H, W, 3)
    """
    H, W = int(H), int(W)
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H), indexing='xy') # pytorch's meshgrid has indexing='ij'
    i = i.t().to(c2w.device)
    j = j.t().to(c2w.device)
    dirs = torch.stack([(i-K[0][2])/K[0][0], (j-K[1][2])/K[1][1], torch.ones_like(i)], -1).to(c2w.device)
    if format == "OpenGL" or format == "Blender":
        dirs[..., 1:] = -dirs[..., 1:]
    # Rotate ray directions from camera frame to the world frame
    rays_d = c2w[:3, :3].matmul(dirs[..., None]).squeeze(-1)
    if normalize_dir:
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        print(f"do rays_d normalization")
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d

def compose_extrinsic_R_T(R: torch.Tensor, T: torch.Tensor):
    """
    Compose the standard form extrinsic matrix from R and T.
    Batched I/O.
    """
    RT = torch.cat((R, T.unsqueeze(-1)), dim=-1)
    return compose_extrinsic_RT(RT)


def compose_extrinsic_RT(RT: torch.Tensor):
    """
    Compose the standard form extrinsic matrix from RT.
    Batched I/O.
    """
    return torch.cat([
        RT,
        torch.tensor([[[0, 0, 0, 1]]], dtype=RT.dtype, device=RT.device).repeat(RT.shape[0], 1, 1)
        ], dim=1)


def decompose_extrinsic_R_T(E: torch.Tensor):
    """
    Decompose the standard extrinsic matrix into R and T.
    Batched I/O.
    """
    RT = decompose_extrinsic_RT(E)
    return RT[:, :, :3], RT[:, :, 3]


def decompose_extrinsic_RT(E: torch.Tensor):
    """
    Decompose the standard extrinsic matrix into RT.
    Batched I/O.
    """
    return E[:, :3, :]

def get_normalized_camera_intrinsics(intrinsics: torch.Tensor):
    """
    intrinsics: (N, 3, 2), [[fx, fy], [cx, cy], [width, height]]
    Return batched fx, fy, cx, cy
    """
    fx, fy = intrinsics[:, 0, 0], intrinsics[:, 0, 1]
    cx, cy = intrinsics[:, 1, 0], intrinsics[:, 1, 1]
    width, height = intrinsics[:, 2, 0], intrinsics[:, 2, 1]
    fx, fy = fx / width, fy / height
    cx, cy = cx / width, cy / height
    return fx, fy, cx, cy

def camera_normalization_objaverse(normed_dist_to_center, poses: torch.Tensor, ret_transform: bool = False):
    assert normed_dist_to_center is not None
    pivotal_pose = compose_extrinsic_RT(poses[:1])
    dist_to_center = pivotal_pose[:, :3, 3].norm(dim=-1, keepdim=True).item() \
        if normed_dist_to_center == 'auto' else normed_dist_to_center

    # compute camera norm (new version)
    canonical_camera_extrinsics = torch.tensor([[
        [1, 0, 0, 0],
        [0, 0, -1, -dist_to_center],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ]], dtype=torch.float32)
    pivotal_pose_inv = torch.inverse(pivotal_pose)
    camera_norm_matrix = torch.bmm(canonical_camera_extrinsics, pivotal_pose_inv)

    # normalize all views
    poses = compose_extrinsic_RT(poses)
    poses = torch.bmm(camera_norm_matrix.repeat(poses.shape[0], 1, 1), poses)
    poses = decompose_extrinsic_RT(poses)

    if ret_transform:
        return poses, camera_norm_matrix.squeeze(dim=0)
    return poses

def camera_relative_pose_koolai(poses: Float[Tensor, "N 4 4"], ret_transform: bool = False):
    pivotal_pose = poses[:1]
    pivotal_pose_inv = torch.inverse(pivotal_pose)

    # normalize all views
    poses = torch.bmm(pivotal_pose_inv.repeat(poses.shape[0], 1, 1), poses)

    if ret_transform:
        return poses, pivotal_pose_inv.squeeze(dim=0)
    return poses


def build_camera_principle(RT: torch.Tensor, intrinsics: torch.Tensor):
    """
    RT: (N, 3, 4)
    intrinsics: (N, 3, 2), [[fx, fy], [cx, cy], [width, height]]
    """
    fx, fy, cx, cy = get_normalized_camera_intrinsics(intrinsics)
    return torch.cat([
        RT.reshape(-1, 12),
        fx.unsqueeze(-1), fy.unsqueeze(-1), cx.unsqueeze(-1), cy.unsqueeze(-1),
    ], dim=-1)


def build_camera_standard(RT: torch.Tensor, intrinsics: torch.Tensor):
    """
    RT: (N, 3, 4)
    intrinsics: (N, 3, 2), [[fx, fy], [cx, cy], [width, height]]
    """
    E = compose_extrinsic_RT(RT)
    fx, fy, cx, cy = get_normalized_camera_intrinsics(intrinsics)
    I = torch.stack([
        torch.stack([fx, torch.zeros_like(fx), cx], dim=-1),
        torch.stack([torch.zeros_like(fy), fy, cy], dim=-1),
        torch.tensor([[0, 0, 1]], dtype=torch.float32, device=RT.device).repeat(RT.shape[0], 1),
    ], dim=1)
    return torch.cat([
        E.reshape(-1, 16),
        I.reshape(-1, 9),
    ], dim=-1)
    
def build_spherical_camera_standard(extrinsic: torch.Tensor):
    """
    extrinsic: (N, 4, 4)
    """
    E = extrinsic
    return E.reshape(-1, 16)

def project_points(points: torch.Tensor, extrinsic: torch.Tensor):
    """ project 3D points from world to camera space, or vice versa
    points: (B, N, 3)
    extrinsic: (B, 4, 4)
    """
    points = torch.cat([points, torch.ones_like(points[:, :, :1])], dim=-1)
    points = torch.bmm(extrinsic, points.permute(0, 2, 1)).permute(0, 2, 1)
    points = points[:,:,:3] / points[:, :, 3:]
    return points

def cvt_to_perspective_pointcloud(rgb_image: torch.Tensor,
                                  depth_image: torch.Tensor,
                                  depth_scale: float = 1.0,
                                  hfov_rad: float = 0.5*np.pi,
                                  wfov_rad: float = 0.5*np.pi,):
    """
    rgb: (3, H, W)
    depth: (1, H, W)

    Returns dict with 'points' (N, 3) and 'colors' (N, 3) numpy arrays.
    """
    C, H, W = depth_image.shape
    K = torch.tensor([
                [1 / np.tan(wfov_rad / 2.), 0., 0., 0.],
                [0., 1 / np.tan(hfov_rad / 2.), 0., 0.],
                [0., 0.,  1, 0],
                [0., 0., 0, 1]]).float().to(depth_image.device)

    # Now get an approximation for the true world coordinates -- see if they make sense
    # [-1, 1] for x and [-1, -1] for y as array indexing is y-down
    ys, xs = torch.meshgrid(torch.linspace(-1,1,H), torch.linspace(-1,1,W), indexing='ij')
    xs = xs.reshape(1,H,W).to(depth_image.device)
    ys = ys.reshape(1,H,W).to(depth_image.device)

    # Unproject
    # positive depth as the camera looks along Z
    depth = depth_image / depth_scale
    xys = torch.cat([xs * depth , ys * depth, depth, torch.ones_like(depth)], dim=0)
    xys = xys.reshape(4, -1)
    points = torch.inverse(K) @ xys

    colors = rgb_image.permute(1, 2, 0).reshape(-1, 3).cpu().numpy()
    colors = np.clip(colors, 0, 1.0)
    points = points.permute(1, 0).cpu().numpy()[:, :3]
    return {"points": points, "colors": colors}

import math
def focus_of_attention(poses: Float[Tensor, "*num_poses 4 4"], initial_focus: Float[Tensor, "3"]) -> Float[Tensor, "3"]:
    """Compute the focus of attention of a set of cameras. Only cameras
    that have the focus of attention in front of them are considered.

     Args:
        poses: The poses to orient.
        initial_focus: The 3D point views to decide which cameras are initially activated.

    Returns:
        The 3D position of the focus of attention.
    """
    # References to the same method in third-party code:
    # https://github.com/google-research/multinerf/blob/1c8b1c552133cdb2de1c1f3c871b2813f6662265/internal/camera_utils.py#L145
    # https://github.com/bmild/nerf/blob/18b8aebda6700ed659cb27a0c348b737a5f6ab60/load_llff.py#L197
    active_directions = -poses[:, :3, 2:3]
    active_origins = poses[:, :3, 3:4]
    # initial value for testing if the focus_pt is in front or behind
    focus_pt = initial_focus
    # Prune cameras which have the current have the focus_pt behind them.
    active = torch.sum(active_directions.squeeze(-1) * (focus_pt - active_origins.squeeze(-1)), dim=-1) > 0
    done = False
    # We need at least two active cameras, else fallback on the previous solution.
    # This may be the "poses" solution if no cameras are active on first iteration, e.g.
    # they are in an outward-looking configuration.
    while torch.sum(active.int()) > 1 and not done:
        active_directions = active_directions[active]
        active_origins = active_origins[active]
        # https://en.wikipedia.org/wiki/Line–line_intersection#In_more_than_two_dimensions
        m = torch.eye(3) - active_directions * torch.transpose(active_directions, -2, -1)
        mt_m = torch.transpose(m, -2, -1) @ m
        focus_pt = torch.linalg.inv(mt_m.mean(0)) @ (mt_m @ active_origins).mean(0)[:, 0]
        active = torch.sum(active_directions.squeeze(-1) * (focus_pt - active_origins.squeeze(-1)), dim=-1) > 0
        if active.all():
            # the set of active cameras did not change, so we're done.
            done = True
    return focus_pt

def rotation_matrix_between(a: Float[Tensor, "3"], b: Float[Tensor, "3"]) -> Float[Tensor, "3 3"]:
    """Compute the rotation matrix that rotates vector a to vector b.

    Args:
        a: The vector to rotate.
        b: The vector to rotate to.
    Returns:
        The rotation matrix.
    """
    a = a / torch.linalg.norm(a)
    b = b / torch.linalg.norm(b)
    v = torch.linalg.cross(a, b)  # Axis of rotation.

    # Handle cases where `a` and `b` are parallel.
    eps = 1e-6
    if torch.sum(torch.abs(v)) < eps:
        x = torch.tensor([1.0, 0, 0]) if abs(a[0]) < eps else torch.tensor([0, 1.0, 0])
        v = torch.linalg.cross(a, x)

    v = v / torch.linalg.norm(v)
    skew_sym_mat = torch.Tensor(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ]
    )
    theta = torch.acos(torch.clip(torch.dot(a, b), -1, 1))

    # Rodrigues rotation formula. https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    return torch.eye(3) + torch.sin(theta) * skew_sym_mat + (1 - torch.cos(theta)) * (skew_sym_mat @ skew_sym_mat)

def auto_orient_and_center_poses(
    poses: Float[Tensor, "*num_poses 4 4"],
    method: Literal["pca", "up", "vertical", "none"] = "up",
    center_method: Literal["poses", "focus", "none"] = "poses",
) -> Tuple[Float[Tensor, "*num_poses 3 4"], Float[Tensor, "3 4"]]:
    """Orients and centers the poses.

    We provide three methods for orientation:

    - pca: Orient the poses so that the principal directions of the camera centers are aligned
        with the axes, Z corresponding to the smallest principal component.
        This method works well when all of the cameras are in the same plane, for example when
        images are taken using a mobile robot.
    - up: Orient the poses so that the average up vector is aligned with the z axis.
        This method works well when images are not at arbitrary angles.
    - vertical: Orient the poses so that the Z 3D direction projects close to the
        y axis in images. This method works better if cameras are not all
        looking in the same 3D direction, which may happen in camera arrays or in LLFF.

    There are two centering methods:

    - poses: The poses are centered around the origin.
    - focus: The origin is set to the focus of attention of all cameras (the
        closest point to cameras optical axes). Recommended for inward-looking
        camera configurations.

    Args:
        poses: The poses to orient.
        method: The method to use for orientation.
        center_method: The method to use to center the poses.

    Returns:
        Tuple of the oriented poses and the transform matrix.
    """

    origins = poses[..., :3, 3]

    mean_origin = torch.mean(origins, dim=0)
    translation_diff = origins - mean_origin

    if center_method == "poses":
        translation = mean_origin
    elif center_method == "focus":
        translation = focus_of_attention(poses, mean_origin)
    elif center_method == "none":
        translation = torch.zeros_like(mean_origin)
    else:
        raise ValueError(f"Unknown value for center_method: {center_method}")

    if method == "pca":
        _, eigvec = torch.linalg.eigh(translation_diff.T @ translation_diff)
        eigvec = torch.flip(eigvec, dims=(-1,))

        if torch.linalg.det(eigvec) < 0:
            eigvec[:, 2] = -eigvec[:, 2]

        transform = torch.cat([eigvec, eigvec @ -translation[..., None]], dim=-1)
        oriented_poses = transform @ poses

        if oriented_poses.mean(dim=0)[2, 1] < 0:
            oriented_poses[1:3, :] = -1 * oriented_poses[1:3, :]
            transform[1:3, :] = -1 * transform[1:3, :]
    elif method in ("up", "vertical"):
        up = torch.mean(poses[:, :3, 1], dim=0)
        up = up / torch.linalg.norm(up)
        if method == "vertical":
            # If cameras are not all parallel (e.g. not in an LLFF configuration),
            # we can find the 3D direction that most projects vertically in all
            # cameras by minimizing ||Xu|| s.t. ||u||=1. This total least squares
            # problem is solved by SVD.
            x_axis_matrix = poses[:, :3, 0]
            _, S, Vh = torch.linalg.svd(x_axis_matrix, full_matrices=False)
            # Singular values are S_i=||Xv_i|| for each right singular vector v_i.
            # ||S|| = sqrt(n) because lines of X are all unit vectors and the v_i
            # are an orthonormal basis.
            # ||Xv_i|| = sqrt(sum(dot(x_axis_j,v_i)^2)), thus S_i/sqrt(n) is the
            # RMS of cosines between x axes and v_i. If the second smallest singular
            # value corresponds to an angle error less than 10° (cos(80°)=0.17),
            # this is probably a degenerate camera configuration (typical values
            # are around 5° average error for the true vertical). In this case,
            # rather than taking the vector corresponding to the smallest singular
            # value, we project the "up" vector on the plane spanned by the two
            # best singular vectors. We could also just fallback to the "up"
            # solution.
            if S[1] > 0.17 * math.sqrt(poses.shape[0]):
                # regular non-degenerate configuration
                up_vertical = Vh[2, :]
                # It may be pointing up or down. Use "up" to disambiguate the sign.
                up = up_vertical if torch.dot(up_vertical, up) > 0 else -up_vertical
            else:
                # Degenerate configuration: project "up" on the plane spanned by
                # the last two right singular vectors (which are orthogonal to the
                # first). v_0 is a unit vector, no need to divide by its norm when
                # projecting.
                up = up - Vh[0, :] * torch.dot(up, Vh[0, :])
                # re-normalize
                up = up / torch.linalg.norm(up)

        rotation = rotation_matrix_between(up, torch.Tensor([0, 0, 1]))
        transform = torch.cat([rotation, rotation @ -translation[..., None]], dim=-1)
        oriented_poses = transform @ poses
    elif method == "none":
        transform = torch.eye(4)
        transform[:3, 3] = -translation
        transform = transform[:3, :]
        oriented_poses = transform @ poses
    else:
        raise ValueError(f"Unknown value for method: {method}")

    return oriented_poses, transform

# def unproject_depth(depth_map: Tensor, C2W: Tensor, fxfycxcy: Tensor) -> Tensor:
#     """Unproject depth map to 3D world coordinate.

#     Inputs:
#         - `depth_map`: (B, V, H, W)
#         - `C2W`: (B, V, 4, 4)
#         - `fxfycxcy`: (B, V, 4)

#     Outputs:
#         - `xyz_world`: (B, V, 3, H, W)
#     """
#     device, dtype = depth_map.device, depth_map.dtype
#     B, V, H, W = depth_map.shape

#     depth_map = depth_map.reshape(B*V, H, W).float()
#     C2W = C2W.reshape(B*V, 4, 4).float()
#     fxfycxcy = fxfycxcy.reshape(B*V, 4).float()
#     K = torch.zeros(B*V, 3, 3, dtype=torch.float32, device=device)
#     K[:, 0, 0] = fxfycxcy[:, 0]
#     K[:, 1, 1] = fxfycxcy[:, 1]
#     K[:, 0, 2] = fxfycxcy[:, 2]
#     K[:, 1, 2] = fxfycxcy[:, 3]
#     K[:, 2, 2] = 1

#     y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")  # OpenCV/COLMAP camera convention
#     y = y.to(device).unsqueeze(0).repeat(B*V, 1, 1) / (H-1)
#     x = x.to(device).unsqueeze(0).repeat(B*V, 1, 1) / (W-1)
#     # NOTE: To align with `plucker_ray(bug=False)`, should be:
#     # y = (y.to(device).unsqueeze(0).repeat(B*V, 1, 1) + 0.5) / H
#     # x = (x.to(device).unsqueeze(0).repeat(B*V, 1, 1) + 0.5) / W
#     xyz_map = torch.stack([x, y, torch.ones_like(x)], axis=-1) * depth_map[..., None]
#     xyz = xyz_map.view(B*V, -1, 3)

#     # Get point positions in camera coordinate
#     xyz = torch.matmul(xyz, torch.transpose(torch.inverse(K), 1, 2))
#     xyz_map = xyz.view(B*V, H, W, 3)

#     # Transform pts from camera to world coordinate
#     xyz_homo = torch.ones((B*V, H, W, 4), device=device)
#     xyz_homo[..., :3] = xyz_map
#     xyz_world = torch.bmm(C2W, xyz_homo.reshape(B*V, -1, 4).permute(0, 2, 1))[:, :3, ...].to(dtype)  # (B*V, 3, H*W)
#     xyz_world = xyz_world.reshape(B, V, 3, H, W)
#     return xyz_world

def unproject_depth(depth_map: Tensor, C2W: Tensor, K: Tensor) -> Tensor:
    """Unproject depth map to 3D world coordinate.

    Inputs:
        - `depth_map`: (V, H, W)
        - `C2W`: (V, 4, 4)
        - `K`: (3, 3)

    Outputs:
        - `xyz_world`: (V, 3, H, W)
    """
    device, dtype = depth_map.device, depth_map.dtype
    V, _, H, W = depth_map.shape

    depth_map = depth_map.permute(0, 2,3,1).float()
    # C2W = C2W.reshape(V, 4, 4).float()
    if len(K.shape) != 3:
        K = K.view(1, 3, 3)
    K = K.repeat(V, 1, 1).float()

    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")  # OpenCV/COLMAP camera convention
    y = y.to(device).unsqueeze(0).repeat(V, 1, 1)
    x = x.to(device).unsqueeze(0).repeat(V, 1, 1)
    # NOTE: To align with `plucker_ray(bug=False)`, should be:
    # y = (y.to(device).unsqueeze(0).repeat(B*V, 1, 1) + 0.5) / H
    # x = (x.to(device).unsqueeze(0).repeat(B*V, 1, 1) + 0.5) / W
    xyz_map = torch.stack([x, y, torch.ones_like(x)], axis=-1) * depth_map
    xyz = xyz_map.view(V, -1, 3)

    # Get point positions in camera coordinate
    xyz = torch.matmul(xyz, torch.transpose(torch.inverse(K), 1, 2))
    xyz_map = xyz.view(V, H, W, 3)

    # Transform pts from camera to world coordinate
    xyz_homo = torch.ones((V, H, W, 4), device=device)
    xyz_homo[..., :3] = xyz_map
    xyz_world = torch.bmm(C2W, xyz_homo.reshape(V, -1, 4).permute(0, 2, 1))[:, :3, ...].to(dtype)  # (B*V, 3, H*W)
    xyz_world = xyz_world.reshape(V, 3, H, W)
    return xyz_world

def opengl_to_opencv(c2w):
    transform = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    if isinstance(c2w, torch.Tensor):
        transform = torch.Tensor(transform).to(c2w)
    c2w[..., :3, :3] @= transform
    return c2w