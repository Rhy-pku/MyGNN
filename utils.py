import os
import numpy as np
import torch


# def spherical_to_unit_cube(points):
#     """Convert cartesian coordinates on the 4D unit sphere to coordinates in [0, 1]^3 using PyTorch."""
#     x = points[:, 0]
#     y = points[:, 1]
#     z = points[:, 2]
#     w = points[:, 3]

#     theta1 = torch.atan2(y, x)
#     theta1 = torch.where(theta1 < 0, theta1 + 2 * torch.pi, theta1)

#     r = torch.sqrt(x**2 + y**2)
#     theta2 = torch.atan2(z, r)
#     theta2 = torch.where(theta2 < 0, theta2 + 2 * torch.pi, theta2)

#     phi = torch.acos(w)

#     a = theta1 / (2 * torch.pi)
#     b = (torch.cos(theta2) + 1) / 2
#     c = (torch.cos(phi) + 1) / 2

#     converted_points = torch.stack([a, b, c], dim=1)
#     return converted_points

# def unit_cube_to_spherical(points):
#     """Convert coordinates in [0, 1]^3 to cartesian coordinates on the 4D unit sphere."""
#     points = points.detach().cpu().numpy()  # 确保转换为 NumPy 数组
#     spherical_points = []
#     for point in points:
#         a, b, c = point
#         theta1 = a * 2 * np.pi
#         theta2 = np.arccos(2 * b - 1)
#         phi = np.arccos(2 * c - 1)

#         x = np.cos(theta1) * np.sin(theta2)
#         y = np.sin(theta1) * np.sin(theta2)
#         z = np.cos(theta2)
#         w = np.cos(phi)

#         spherical_points.append([x, y, z, w])
#     return np.array(spherical_points)

def normalize_vectors(vectors):
    """
    将四维向量数组转换为四维单位球面上的点。
    
    Args:
    vectors (torch.Tensor): 输入的四维向量张量，形状为 (n, 4)。
    
    Returns:
    torch.Tensor: 归一化后的四维向量张量，形状为 (n, 4)。
    """
    norms = torch.norm(vectors, dim=1, keepdim=True) + 1e-10  # 防止除零错误
    normalized_vectors = vectors / norms
    return normalized_vectors

# def torch_l2_discrepancy(points):
#     """
#     计算点集的L2差异性。
    
#     Args:
#     points (torch.Tensor): 输入的四维单位球面上的点，形状为 (n, 4)。
    
#     Returns:
#     tuple: L2差异性和 sum1 值。
#     """
#     sphere_points = normalize_vectors(points)
#     cube_points = spherical_to_unit_cube(sphere_points)
    
#     n, d = cube_points.shape
#     cube_points = cube_points.to(points.device)

#     sum1 = torch.zeros(1, device=points.device, requires_grad=True)
#     sum2 = torch.zeros(1, device=points.device, requires_grad=True)

#     for i in range(n):
#         for j in range(n):
#             prod = torch.ones(1, device=points.device, requires_grad=True)
#             for k in range(d):
#                 prod = prod * (1 - torch.max(cube_points[i, k], cube_points[j, k]))
#             sum1 = sum1 + prod

#     for i in range(n):
#         prod = torch.ones(1, device=points.device, requires_grad=True)
#         for k in range(d):
#             prod = prod * (1 - cube_points[i, k]**2) / 2
#         sum2 = sum2 + prod

#     c_d = (1.0 / 3**d)
#     l2_disc = c_d - ((2.0 / n) * sum2) + (1.0 / n**2) * sum1 

#     return l2_disc, sum1  # 返回 sum1 以确保梯度计算



def cos_losses(points, threshold=0.5):
    n = len(points)
    norms = torch.norm(points, dim=1, keepdim=True)  # 计算每个向量的范数
    normalized_points = points / norms  # 规范化每个向量
    cos_similarity_matrix = torch.mm(normalized_points, normalized_points.t())  # 计算余弦相似度矩阵
    mask = torch.triu(torch.ones(n, n, device=points.device), diagonal=1)
    cos_similarity_above_threshold = torch.where(cos_similarity_matrix > threshold, cos_similarity_matrix, torch.tensor(0.0, device=points.device))
    cos_similarity_above_threshold = cos_similarity_above_threshold * mask
    loss = torch.sum(torch.max(torch.tensor(0.0, device=points.device), cos_similarity_above_threshold - threshold) ** 2 * 1000000)
    return loss


# def cos_loss(points, threshold=0.5):
#     cnt = 0
#     # epsilon = 1e-30  # 防止除零的小正值
#     n = len(points)
#     loss = torch.tensor(0.0, device=points.device, requires_grad=True)
#     for i in range(n):
#         cnt = 0
#         for j in range(0, n):
#             if i == j:
#                 continue
#             norm_i = torch.norm(points[i]) 
#             norm_j = torch.norm(points[j]) 
#             cos_similarity = torch.dot(points[i], points[j]) / (norm_i * norm_j)
#             # if(cos_similarity < 0.5):
#             #     loss = loss - 0.1
#             if 1:
#                 if(cos_similarity > 0.5):
#                     loss = loss + torch.max(torch.tensor(0.0, device=points.device), cos_similarity - threshold)*1000000#可以考虑平方

#     return loss

# def cos_losses_best(points, threshold=0.5):
#     cnt = 0
#     # epsilon = 1e-30  # 防止除零的小正值
#     n = len(points)
#     loss = torch.tensor(0.0, device=points.device, requires_grad=True)
#     for i in range(n):
#         cnt = 0
#         for j in range(0, n):
#             if i == j:
#                 continue
#             norm_i = torch.norm(points[i]) 
#             norm_j = torch.norm(points[j]) 
#             cos_similarity = torch.dot(points[i], points[j]) / (norm_i * norm_j)
#             # if(cos_similarity < 0.5):
#             #     loss = loss - 0.1
#             if cos_similarity > 0.5:
#                 loss = loss + (cos_similarity - threshold)**2
#             elif cos_similarity > 0.4:
#                 loss = loss + (cos_similarity - threshold)**2
#     return loss


def save_spheres(spheres, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    np.save(filename, spheres)

def compute_edges(points, threshold=0.15):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    normalized_points = points / norms
    cos_similarity_matrix = np.dot(normalized_points, normalized_points.T)
    edges = np.argwhere(cos_similarity_matrix > threshold)
    edges = edges[edges[:, 0] != edges[:, 1]]  # Remove self-loops
    return edges.T if edges.size > 0 else np.array([], dtype=np.int64).reshape(2, 0)

