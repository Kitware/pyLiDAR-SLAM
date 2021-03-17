import numpy as np
import torch

from slam.common.utils import assert_debug, check_sizes


def Rx(phi):
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(phi), -np.sin(phi)],
        [0.0, np.sin(phi), np.cos(phi)]])


def torch_rx(cos_phi, sin_phi):
    """
    >>> check_sizes(torch_rx(torch.randn(6), torch.randn(6)), [6, 3, 3])
    """
    check_sizes(cos_phi, [-1])
    b = cos_phi.size(0)
    check_sizes(sin_phi, [b])

    rot = torch.zeros(b, 3, 3, device=cos_phi.device, dtype=cos_phi.dtype)
    rot[:, 0, 0] = 1.0
    rot[:, 1, 1] = cos_phi
    rot[:, 2, 2] = cos_phi
    rot[:, 1, 2] = - sin_phi
    rot[:, 2, 1] = sin_phi
    return rot


def JRx(phi):
    return np.array([
        [0.0, 0.0, 0.0],
        [0.0, -np.sin(phi), -np.cos(phi)],
        [0.0, np.cos(phi), -np.sin(phi)]
    ])


def torch_jac_rx(cos_phi, sin_phi):
    check_sizes(cos_phi, [-1])
    b = cos_phi.size(0)
    check_sizes(sin_phi, [b])

    rot = torch.zeros(b, 3, 3, device=cos_phi.device, dtype=cos_phi.dtype)
    rot[:, 0, 0] = 0.0
    rot[:, 1, 1] = -sin_phi
    rot[:, 2, 2] = -sin_phi
    rot[:, 1, 2] = -cos_phi
    rot[:, 2, 1] = cos_phi
    return rot


def Ry(theta):
    return np.array([
        [np.cos(theta), 0.0, np.sin(theta)],
        [0.0, 1.0, 0.0],
        [-np.sin(theta), 0.0, np.cos(theta)]])


def torch_ry(cos_phi, sin_phi):
    """
    >>> check_sizes(torch_ry(torch.randn(6), torch.randn(6)), [6, 3, 3])
    """
    check_sizes(cos_phi, [-1])
    b = cos_phi.size(0)
    check_sizes(sin_phi, [b])

    rot = torch.zeros(b, 3, 3, device=cos_phi.device, dtype=cos_phi.dtype)
    rot[:, 0, 0] = cos_phi
    rot[:, 1, 1] = 1.0
    rot[:, 2, 2] = cos_phi
    rot[:, 0, 2] = sin_phi
    rot[:, 2, 0] = -sin_phi
    return rot


def JRy(theta):
    return np.array([
        [-np.sin(theta), 0.0, np.cos(theta)],
        [0.0, 0.0, 0.0],
        [-np.cos(theta), 0.0, -np.sin(theta)]
    ])


def torch_jac_ry(cos_theta, sin_theta):
    check_sizes(cos_theta, [-1])
    b = cos_theta.size(0)
    check_sizes(sin_theta, [b])

    rot = torch.zeros(b, 3, 3, device=cos_theta.device, dtype=cos_theta.dtype)
    rot[:, 0, 0] = - sin_theta
    rot[:, 2, 2] = - sin_theta
    rot[:, 0, 2] = cos_theta
    rot[:, 2, 0] = -cos_theta
    return rot


def Rz(psi):
    return np.array([
        [np.cos(psi), - np.sin(psi), 0.0],
        [np.sin(psi), np.cos(psi), 0.0],
        [0.0, 0.0, 1.0]])


def torch_rz(cos_psi, sin_psi):
    check_sizes(cos_psi, [-1])
    b = cos_psi.size(0)
    check_sizes(sin_psi, [b])

    rot = torch.zeros(b, 3, 3, device=cos_psi.device, dtype=cos_psi.dtype)
    rot[:, 0, 0] = cos_psi
    rot[:, 1, 1] = cos_psi
    rot[:, 2, 2] = 1.0
    rot[:, 0, 1] = - sin_psi
    rot[:, 1, 0] = sin_psi
    return rot


def JRz(psi):
    return np.array([[-np.sin(psi), -np.cos(psi), 0.0],
                     [np.cos(psi), -np.sin(psi), 0.0],
                     [0.0, 0.0, 0.0]])


def torch_jac_rz(cos_psi, sin_psi):
    check_sizes(cos_psi, [-1])
    b = cos_psi.size(0)
    check_sizes(sin_psi, [b])

    rot = torch.zeros(b, 3, 3, device=cos_psi.device, dtype=cos_psi.dtype)
    rot[:, 0, 0] = -sin_psi
    rot[:, 1, 1] = -sin_psi
    rot[:, 0, 1] = - cos_psi
    rot[:, 1, 0] = cos_psi
    return rot


def euler_to_mat(angles, convention="xyz"):
    assert_debug(convention == "xyz")
    ex, ey, ez = angles
    return Rz(ez).dot(Ry(ey).dot(Rx(ex)))


def torch_euler_to_mat(angles, convention="xyz"):
    assert_debug(convention == "xyz")
    torch_cos = torch.cos(angles)
    torch_sin = torch.sin(angles)
    return torch_rz(torch_cos[:, 2], torch_sin[:, 2]) @ \
           torch_ry(torch_cos[:, 1], torch_sin[:, 1]) @ \
           torch_rx(torch_cos[:, 0], torch_sin[:, 0])


def euler_jacobian(angles, convention="xyz"):
    assert_debug(convention == "xyz")
    ex, ey, ez = angles
    rot_z = Rz(ez)
    rot_y = Ry(ey)
    rot_x = Rx(ex)
    return np.vstack([
        np.expand_dims(rot_z.dot(rot_y)._dot(JRx(ex)), 0),
        np.expand_dims(rot_z.dot(JRy(ey))._dot(rot_x), 0),
        np.expand_dims(JRz(ez).dot(rot_y)._dot(rot_x), 0)
    ])


def torch_euler_jacobian(angles, convention="xyz"):
    """
    Returns
    -------
        [B, 3, 3, 3]

    """
    assert_debug(convention == "xyz")
    check_sizes(angles, [-1, 3])
    torch_cos = torch.cos(angles)
    torch_sin = torch.sin(angles)

    rot_z = torch_rz(torch_cos[:, 2], torch_sin[:, 2])
    rot_y = torch_ry(torch_cos[:, 1], torch_sin[:, 1])
    rot_x = torch_rx(torch_cos[:, 0], torch_sin[:, 0])
    return torch.cat([
        (rot_z @ rot_y @ torch_jac_rx(torch_cos[:, 0], torch_sin[:, 0])).unsqueeze(1),
        (rot_z @ torch_jac_ry(torch_cos[:, 1], torch_sin[:, 1]) @ rot_x).unsqueeze(1),
        (torch_jac_rz(torch_cos[:, 2], torch_sin[:, 2]) @ rot_y @ rot_x).unsqueeze(1)], dim=1)


def torch_pose_matrix_jacobian_euler(pose_params, convention="xyz"):
    """

    Parameters
    ----------
    pose_params :
        The pose parameters with euler convention [B, 6]
        [B, :3] the translation parameters
        [B, 3:] the euler angle parameters
    convention :
        The euler convention of the params

    Returns
    -------
    torch.Tensor [B, 6, 4, 4]

    """
    assert_debug(convention == "xyz")
    check_sizes(pose_params, [-1, 6])
    n = pose_params.size(0)
    euler_jac = torch_euler_jacobian(pose_params[:, 3:], convention)
    pose_matrix_jac = torch.zeros(n, 6, 4, 4,
                                  dtype=pose_params.dtype,
                                  device=pose_params.device)
    pose_matrix_jac[:, 0, 0, 3] = 1.0
    pose_matrix_jac[:, 1, 1, 3] = 1.0
    pose_matrix_jac[:, 2, 2, 3] = 1.0

    pose_matrix_jac[:, 3:, :3, :3] = euler_jac
    return pose_matrix_jac


def is_rotation_matrix(rot, eps=1.e-5):
    if isinstance(rot, np.ndarray):
        rot_t = np.transpose(rot)
        rot_t_rot = rot.dot(rot_t)
        id3 = np.eye(3, dtype=rot.dtype)
        n = np.linalg.norm(id3 - rot_t_rot)
        return n < eps
    elif isinstance(rot, torch.Tensor):
        check_sizes(rot, [-1, 3, 3])
        rot_t = rot.transpose(1, 2)
        rot_t_rot = rot @ rot_t
        id3 = torch.eye(3, dtype=rot.dtype, device=rot.device)
        n = (id3 - rot_t_rot).abs().max()
        return n < eps
    else:
        raise NotImplementedError("")


def mat_to_euler(rot, convention="xyz", eps=1.e-6):
    assert_debug(convention == "xyz")
    # assert_debug(is_rotation_matrix(rot))
    sy = np.sqrt(rot[0, 0] * rot[0, 0] + rot[1, 0] * rot[1, 0])
    singular = sy < eps
    if not singular:
        x = np.arctan2(rot[2, 1], rot[2, 2])
        y = np.arctan2(-rot[2, 0], sy)
        z = np.arctan2(rot[1, 0], rot[0, 0])
    else:
        x = np.arctan2(-rot[1, 2], rot[1, 1])
        y = np.arctan2(-rot[2, 0], sy)
        z = 0
    return np.array([x, y, z])


def torch_mat_to_euler(rot, convention="xyz", eps=1.e-6):
    assert_debug(convention == "xyz")
    # assert_debug(is_rotation_matrix(rot), f" Not a rotation matrix {rot}")
    sy = torch.sqrt(rot[:, 0, 0] * rot[:, 0, 0] + rot[:, 1, 0] * rot[:, 1, 0])
    is_singular = sy < eps

    x_ns = torch.atan2(rot[:, 2, 1], rot[:, 2, 2])
    y_ns = torch.atan2(-rot[:, 2, 0], sy)
    z_ns = torch.atan2(rot[:, 1, 0], rot[:, 0, 0])

    x_s = torch.atan2(-rot[:, 1, 2], rot[:, 1, 1])
    y_s = torch.atan2(-rot[:, 2, 0], sy)
    z_s = 0

    return torch.cat([
        (x_ns * ~is_singular + x_s * is_singular).unsqueeze(1),
        (y_ns * ~is_singular + y_s * is_singular).unsqueeze(1),
        (z_ns * ~is_singular + z_s * is_singular).unsqueeze(1)], dim=1)


def torch_mat_to_euler2(rot, convention="xyz", eps=1.e-6):
    assert_debug(convention == "xyz")
    # assert_debug(is_rotation_matrix(rot))

    thetas1 = -torch.asin(rot[:, 2, 0])
    thetas2 = np.pi - thetas1

    c_theta1 = torch.cos(thetas1)
    c_theta2 = torch.cos(thetas2)

    psi1 = torch.atan2(rot[:, 2, 1] / c_theta1, rot[:, 2, 2] / c_theta1)
    psi2 = torch.atan2(rot[:, 2, 1] / c_theta2, rot[:, 2, 2] / c_theta2)

    phi1 = torch.atan2(rot[:, 1, 0] / c_theta1, rot[:, 0, 0] / c_theta1)
    phi2 = torch.atan2(rot[:, 1, 0] / c_theta2, rot[:, 0, 0] / c_theta2)

    # Mask for edge cases
    mask_r20_1 = (rot[:, 2, 0] - 1).abs() < eps  # R[2, 0] == 1
    mask_r20_m1 = (rot[:, 2, 0] + 1).abs() < eps  # R[2, 0] == -1

    phi_edge = torch.zeros_like(phi1)
    thetas_1 = 0.5 * np.pi * mask_r20_m1
    psi_1 = phi_edge + torch.atan2(rot[:, 0, 1], rot[:, 0, 2])
    thetas_m1 = - 0.5 * np.pi * mask_r20_1
    psi_m1 = -phi_edge + torch.atan2(-rot[:, 0, 1], -rot[:, 0, 2])

    mask = ~mask_r20_m1 * ~mask_r20_1

    phi1 = mask * phi1 + ~mask * phi_edge
    psi1 = mask * psi1 + mask_r20_1 * psi_1 + mask_r20_m1 * psi_m1
    thetas1 = mask * thetas1 + mask_r20_1 * thetas_1 + mask_r20_m1 * thetas_m1

    phi2 = mask * phi2 + ~mask * phi_edge
    psi2 = mask * psi2 + mask_r20_1 * psi_1 + mask_r20_m1 * psi_m1
    thetas2 = mask * thetas2 + mask_r20_1 * thetas_1 + mask_r20_m1 * thetas_m1

    angles1 = torch.cat([psi1.unsqueeze(1), thetas1.unsqueeze(1), phi1.unsqueeze(1)], dim=1)
    angles2 = torch.cat([psi2.unsqueeze(1), thetas2.unsqueeze(1), phi2.unsqueeze(1)], dim=1)

    mask_1 = angles1.abs().norm(dim=1, keepdim=True) < angles2.abs().norm(dim=1, keepdim=True)
    angles = angles1 * mask_1 + angles2 * ~mask_1
    return angles
