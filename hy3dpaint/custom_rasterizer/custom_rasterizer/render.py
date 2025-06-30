# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import custom_rasterizer_kernel
import torch

import torch.nn.functional as F

def rasterize_image_torch(V, F, D, width, height, occlusion_truncation, use_depth_prior):
    device = V.device
    num_faces = F.shape[0]
    MAXINT = 1 << 18
    max_token = (MAXINT * MAXINT) + (MAXINT - 1)
    
    # 初始化zbuffer和输出张量
    zbuffer = torch.full((height * width,), max_token, dtype=torch.int64, device=device)
    findices = torch.zeros((height, width), dtype=torch.int32, device=device)
    barycentric = torch.zeros((height, width, 3), dtype=torch.float32, device=device)
    
    # 投影顶点到屏幕空间
    def project_vertex(v):
        x = (v[0] / v[3] * 0.5 + 0.5) * (width - 1) + 0.5
        y = (0.5 + 0.5 * v[1] / v[3]) * (height - 1) + 0.5
        z = v[2] / v[3] * 0.49999 + 0.5
        return torch.stack([x, y, z])
    
    # 计算重心坐标
    def barycentric_coords(v0, v1, v2, p):
        v0_xy, v1_xy, v2_xy = v0[:2], v1[:2], v2[:2]
        denom = (v1_xy[1] - v2_xy[1]) * (v0_xy[0] - v2_xy[0]) + (v2_xy[0] - v1_xy[0]) * (v0_xy[1] - v2_xy[1])
        w0 = ((v1_xy[1] - v2_xy[1]) * (p[0] - v2_xy[0]) + (v2_xy[0] - v1_xy[0]) * (p[1] - v2_xy[1])) / denom
        w1 = ((v2_xy[1] - v0_xy[1]) * (p[0] - v2_xy[0]) + (v0_xy[0] - v2_xy[0]) * (p[1] - v2_xy[1])) / denom
        w2 = 1 - w0 - w1
        return torch.stack([w0, w1, w2])
    
    # 遍历所有三角形
    for f_idx in range(num_faces):
        v0_idx, v1_idx, v2_idx = F[f_idx]
        v0 = V[v0_idx]
        v1 = V[v1_idx]
        v2 = V[v2_idx]
        
        # 投影顶点
        vt0 = project_vertex(v0)
        vt1 = project_vertex(v1)
        vt2 = project_vertex(v2)
        
        # 计算三角形边界框
        x_min = torch.floor(torch.min(torch.stack([vt0[0], vt1[0], vt2[0]]))).int()
        x_max = torch.ceil(torch.max(torch.stack([vt0[0], vt1[0], vt2[0]]))).int()
        y_min = torch.floor(torch.min(torch.stack([vt0[1], vt1[1], vt2[1]]))).int()
        y_max = torch.ceil(torch.max(torch.stack([vt0[1], vt1[1], vt2[1]]))).int()
        
        # 限制在图像范围内
        x_min = torch.clamp(x_min, 0, width - 1)
        x_max = torch.clamp(x_max, 0, width - 1)
        y_min = torch.clamp(y_min, 0, height - 1)
        y_max = torch.clamp(y_max, 0, height - 1)
        
        # 遍历边界框内的所有像素
        for y in range(y_min, y_max + 1):
            for x in range(x_min, x_max + 1):
                pixel_idx = y * width + x
                p = torch.tensor([x + 0.5, y + 0.5], device=device)
                
                # 计算重心坐标
                bary = barycentric_coords(vt0, vt1, vt2, p)
                
                # 检查像素是否在三角形内
                if (bary >= 0).all():
                    # 计算深度值
                    depth = torch.dot(bary, torch.stack([vt0[2], vt1[2], vt2[2]]))
                    
                    # 应用深度先验（如果启用）
                    if use_depth_prior and D is not None:
                        depth_thres = D.view(height, width)[y, x] * 0.49999 + 0.5 + occlusion_truncation
                        if depth < depth_thres:
                            continue
                    
                    # 量化深度值
                    z_quantize = int(depth.item() * (2 << 17))
                    token = z_quantize * MAXINT + (f_idx + 1)
                    
                    # 更新zbuffer
                    if token < zbuffer[pixel_idx]:
                        zbuffer[pixel_idx] = token
    
    # 第二遍：计算重心坐标图
    for y in range(height):
        for x in range(width):
            pixel_idx = y * width + x
            token = zbuffer[pixel_idx]
            f_val = token % MAXINT
            
            if f_val == (MAXINT - 1):
                findices[y, x] = 0
                barycentric[y, x] = 0
            else:
                findices[y, x] = f_val
                f_idx = f_val - 1
                
                if f_idx >= 0:
                    v0_idx, v1_idx, v2_idx = F[f_idx]
                    v0 = V[v0_idx]
                    v1 = V[v1_idx]
                    v2 = V[v2_idx]
                    
                    # 投影顶点（仅XY）
                    vt0 = project_vertex(v0)[:2]
                    vt1 = project_vertex(v1)[:2]
                    vt2 = project_vertex(v2)[:2]
                    
                    # 计算重心坐标
                    p = torch.tensor([x + 0.5, y + 0.5], device=device)
                    bary = barycentric_coords(vt0, vt1, vt2, p)
                    
                    # 应用透视校正
                    bary = bary / torch.stack([v0[3], v1[3], v2[3]])
                    bary /= bary.sum()
                    
                    barycentric[y, x] = bary
    
    return findices, barycentric

def rasterize(pos, tri, resolution, clamp_depth=torch.zeros(0), use_depth_prior=0):
    assert pos.device == tri.device
    pos_cpu = pos[0].cpu()
    tri_cpu = tri.cpu()
    findices, barycentric = custom_rasterizer_kernel.rasterize_image(
        pos_cpu, tri_cpu, clamp_depth, resolution[1], resolution[0], 1e-6, use_depth_prior
    )
    findices = findices.to(pos.device)
    barycentric = barycentric.to(pos.device)
    # findices, barycentric = rasterize_image_torch(
    #     pos[0], tri, clamp_depth, resolution[1], resolution[0], 1e-6, use_depth_prior
    # )
    return findices, barycentric


def interpolate(col, findices, barycentric, tri):
    f = findices - 1 + (findices == 0)
    vcol = col[0, tri.long()[f.long()]]
    result = barycentric.view(*barycentric.shape, 1) * vcol
    result = torch.sum(result, axis=-2)
    return result.view(1, *result.shape)
