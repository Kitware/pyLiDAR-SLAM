from abc import abstractmethod
from typing import Optional, Tuple, Union

import torch
import numpy as np
from slam.common.utils import TensorType

from slam.common.modules import _with_cv2


# ######################################################################################################################
# 3D REGISTRATION
# ######################################################################################################################

def weighted_procrustes(pc_target: TensorType,
                        pc_reference: TensorType,
                        weights: Optional[TensorType] = None) -> TensorType:
    is_torch = isinstance(pc_target, torch.Tensor)
    if is_torch:
        check_tensor(pc_target, [-1, -1, 3])
        check_tensor(pc_reference, [*pc_reference.shape])

        if weights is None:
            weights = torch.ones(pc_target.shape[0], pc_target.shape[1], 1,
                                 dtype=pc_target.dtype, device=pc_target.device)
        avg_weights = weights / weights.sum(dim=1)
    else:
        check_tensor(pc_target, [-1, 3])
        check_tensor(pc_reference, [*pc_target.shape])
        if weights is None:
            weights = np.ones((pc_target.shape[0], 1), dtype=pc_target.dtype)
        avg_weights = weights / weights.sum(axis=0)

    mu_tgt = (pc_target * avg_weights).sum(**(dict(dim=1) if is_torch else dict(axis=0)))
    mu_ref = (pc_reference * avg_weights).sum(**(dict(dim=1) if is_torch else dict(axis=0)))
    if is_torch:
        mu_tgt = mu_tgt.reshape(-1, 1, 3)
        mu_ref = mu_ref.reshape(-1, 1, 3)
    else:
        mu_tgt = mu_tgt.reshape(1, 3)
        mu_ref = mu_ref.reshape(1, 3)

    C_ref_tgt = torch.einsum("bin,bnj->bij", (pc_reference - mu_ref).transpose(1, 2),
                             (pc_target - mu_tgt)).to(torch.float64) if is_torch else \
        np.einsum("in,nj->ij", (pc_reference - mu_ref).T.astype(np.float64), (pc_target - mu_tgt)).astype(np.float64)

    if is_torch:
        U, _, V = C_ref_tgt.svd()
        S = torch.eye(3, device=C_ref_tgt.device, dtype=torch.float64)
        if U.det() * V.det() < 0:
            S[-1, -1] = -1

        mu_tgt = mu_tgt.to(torch.float64)
        mu_ref = mu_ref.to(torch.float64)

        b = C_ref_tgt.shape[0]
        Tr = torch.eye(4, device=C_ref_tgt.device,
                       dtype=torch.float64).reshape(1, 4, 4).repeat(b, 4, 4)
        Tr[:, :3, :3] = torch.einsum("bij,bjk->bik", torch.einsum("bij,jk->bik", U, S), V.permute(0, 2, 1))
        Tr[:, :3, 3] = mu_ref.reshape(b, 3) - (Tr[:, :3, :3] @ mu_tgt.permute(0, 2, 1)).reshape(b, 3)
    else:
        U, _, Vt = np.linalg.svd(C_ref_tgt)
        S = np.eye(3, dtype=np.float64)
        if np.linalg.det(U) * np.linalg.det(Vt) < 0:
            S[-1, -1] = -1

        mu_tgt = mu_tgt.astype(np.float64)
        mu_ref = mu_ref.astype(np.float64)

        Tr = np.eye(4, dtype=np.float64)
        Tr[:3, :3] = U.dot(S).dot(Vt)
        Tr[:3, 3] = mu_ref.reshape(3) - (Tr[:3, :3].dot(mu_tgt.T)).reshape(3)

    return Tr


# ######################################################################################################################
# 2D REGISTRATION
# ######################################################################################################################

if _with_cv2:
    import cv2
    from omegaconf import DictConfig

    from slam.common.utils import check_tensor, assert_debug, TensorType


    class ImageBased2DRegistration:
        """
        Scan registration method using feature based Image Alignment.
        """

        def __init__(self, config: DictConfig):
            super().__init__()
            self.config = config

            # OpenCV algorithms
            features = config.get("features", "orb")
            assert_debug(features in ["orb", "akaze"])
            if features == "akaze":
                self.orb: cv2.Feature2D = cv2.AKAZE_create()
            else:
                self.orb: cv2.Feature2D = cv2.ORB_create()
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

            # Image Construction Parameters
            self.H = self.config.get("im_height", 400)
            self.W = self.config.get("im_width", 400)
            self.inlier_threshold: int = self.config.get("inlier_threshold", 50)
            self._distance_threshold: float = self.config.get("distance_threshold", 2.0)

        @abstractmethod
        def build_image(self, pc: np.ndarray) -> Union[np.ndarray, dict]:
            """Builds the image from the pointcloud (which will be matched by 2D feature based alignment)"""
            raise NotImplementedError("")

        @abstractmethod
        def compute_transorm(self, ref_2d_pts, tgt_2d_pts, **kwargs):
            """Computes the 3D transform from the aligned points"""
            raise NotImplementedError("")

        def compute_features(self, pc: np.ndarray):
            """Projects the pc into the image plane, and compute features and descriptors"""
            image, meta_data = self.build_image(pc)
            # Extract KeyPoints and descriptors
            kpts, desc = self.orb.detectAndCompute(image, None)

            return image, kpts, desc, meta_data

        def compute_inliers(self, ref_pts, tgt_pts):
            """
            Aligns the images using corresponding pair of points (with potentially many outliers)

            By default, the best homography is found
            """
            check_tensor(ref_pts, [-1, 2])
            check_tensor(tgt_pts, [ref_pts.shape[0], 2])
            h, inliers = cv2.findHomography(ref_pts, tgt_pts, cv2.RANSAC, self._distance_threshold)
            return inliers

        def align_2d(self, ref_kpts, ref_desc, tgt_kpts, tgt_desc, ref_img=None, tgt_img=None) -> \
                Tuple[Optional[np.ndarray], np.ndarray, list]:
            """
            Attempts to align the target onto the reference, and if enough inliers are found,

            If succesful, returns the planar transforming the target keypoints into the reference keypoints
            Otherwise, returns None
            """

            matches = self.matcher.match(ref_desc, tgt_desc)
            if len(matches) == 0:
                return None, np.array([], dtype=np.int64), []

            ref_pts = np.array([ref_kpts[m.queryIdx].pt for m in matches])
            tgt_pts = np.array([tgt_kpts[m.trainIdx].pt for m in matches])

            # Find homography to determine the matched pair of points
            inliers = self.compute_inliers(ref_pts, tgt_pts)

            n = inliers.shape[0]
            inliers_indices = np.arange(0, n)[inliers[:, 0].astype(np.bool)]
            inlier_matches = [matches[idx] for idx in inliers_indices]

            ref_pts = ref_pts[inliers_indices]
            tgt_pts = tgt_pts[inliers_indices]

            points = np.concatenate([ref_pts.reshape(-1, 1, 2), tgt_pts.reshape(-1, 1, 2)], axis=1)

            num_inliers = len(inlier_matches)
            if num_inliers < self.inlier_threshold:
                return None, points, inlier_matches

            transform = self.compute_transorm(ref_pts, tgt_pts)

            return transform, points, inlier_matches


    # ------------------------------------------------------------------------------------------------------------------
    class ElevationImageRegistration(ImageBased2DRegistration):
        """2D Feature based registration which estimates the planar motion (x, y, yaw)

        Only relevant for a sensor having "mainly 2D" motion, and can serve as good initialization of this motion
        """

        def __init__(self, config: DictConfig):
            super().__init__(config)
            self.pixel_size: int = self.config.get("pixel_size", 0.4)
            self.z_min: float = self.config.get("z_min", 0.0)
            self.z_max: float = self.config.get("z_max", 5)
            self.sigma: float = self.config.get("sigma", 0.1)
            color_map: str = self.config.get("color_map", "jet")
            self.flip_axis: bool = self.config.get("flip_z_axis", False)
            from matplotlib import cm
            self.color_map = cm.get_cmap(color_map)

        def build_image(self, pc: np.ndarray):
            """Builds an elevation image"""
            image = np.ones((self.H, self.W), dtype=np.float32) * self.z_min

            pc_x = np.round(pc[:, 0] / self.pixel_size + self.H // 2).astype(np.int64)
            pc_y = np.round(pc[:, 1] / self.pixel_size + self.W // 2).astype(np.int64)

            pc_z = pc[:, 2]

            _filter = (0 <= pc_x) * (pc_x < self.H) * (0 <= pc_y) * (pc_y < self.W)

            pc_x = pc_x[_filter]
            pc_y = pc_y[_filter]
            pc_z = pc_z[_filter]
            pc_z = np.clip(pc_z, self.z_min, self.z_max)

            indices = np.argsort(pc_z)[::(1 if self.flip_axis else -1)]

            pc_z = pc_z[indices]
            pc_x = pc_x[indices]
            pc_y = pc_y[indices]

            pixels = np.concatenate([pc_x.reshape(-1, 1), pc_y.reshape(-1, 1)], axis=-1)
            b = np.ascontiguousarray(pixels).view(
                np.dtype((np.void, pixels.dtype.itemsize * pixels.shape[1]))
            )
            values, unique_indices = np.unique(b, return_index=True)

            pc_z = pc_z[unique_indices]
            pixels = pixels[unique_indices]

            thetas = ((pc_z - self.z_min) / (self.z_max - self.z_min)).reshape(-1)
            image[pixels[:, 0], pixels[:, 1]] = thetas
            image = self.color_map(image)[:, :, :3] * 255.0
            image = image.astype(np.uint8)

            # Save in a separate image the original corresponding 3D points for 3D registration
            final_indices = np.arange(0, _filter.shape[0])[_filter][indices][unique_indices]
            pc_points = pc[final_indices]
            pc_image = np.zeros(image.shape, dtype=np.float32)
            pc_image[pixels[:, 0], pixels[:, 1]] = pc_points

            pc_indices = np.ones((image.shape[0], image.shape[1]), dtype=np.int64) * -1
            pc_indices[pixels[:, 0], pixels[:, 1]] = final_indices

            return image, {"points_3D": pc_image, "pc_indices": pc_indices}

        def compute_transorm(self, ref_2d_pts, tgt_2d_pts,
                             meta_data: Optional[dict] = None, **kwargs):
            """Computes the 3D Rigid transform associated to feature based 2D alignment"""
            # Estimate the 2D transform best matching the pair of points
            ref_2d_pts[:, 0] -= self.W // 2
            ref_2d_pts[:, 1] -= self.H // 2
            ref_2d_pts *= self.pixel_size
            ref_2d_pts = ref_2d_pts[:, [1, 0]]  # (row, col) for OpenCV corresponds to (y, x) for pointcloud params

            tgt_2d_pts[:, 0] -= self.W // 2
            tgt_2d_pts[:, 1] -= self.H // 2
            tgt_2d_pts *= self.pixel_size
            tgt_2d_pts = tgt_2d_pts[:, [1, 0]]

            ref_mean = ref_2d_pts.mean(axis=0)
            tgt_mean = tgt_2d_pts.mean(axis=0)
            ref_centered = ref_2d_pts - ref_mean
            tgt_centered = tgt_2d_pts - tgt_mean

            sigma = tgt_centered.T.dot(ref_centered)
            u, d, vt = np.linalg.svd(sigma)

            # Compute The 2D Rotation and translation
            rot2d = vt.T.dot(u.T)
            tr2d = ref_mean - rot2d.dot(tgt_mean)

            # Convert to 3D Relative Pose
            tr = np.eye(4, dtype=np.float32)
            tr[:2, :2] = rot2d
            tr[:2, 3] = tr2d

            return tr
