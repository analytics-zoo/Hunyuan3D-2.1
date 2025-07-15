import sys
sys.path.insert(0, './hy3dshape')
sys.path.insert(0, './hy3dpaint')
import time
from PIL import Image
from hy3dshape.rembg import BackgroundRemover
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

import torch
from torch import nn
import torch.nn.functional as F
from diffusers.models.attention_processor import AttnProcessor2_0, Attention
from typing import Callable, List, Optional, Tuple, Union

try:
    from torchvision_fix import apply_fix
    apply_fix()
except ImportError:
    print("Warning: torchvision_fix module not found, proceeding without compatibility fix")                                      
except Exception as e:
    print(f"Warning: Failed to apply torchvision fix: {e}")

from xpu_convert import convert_to_xpu
convert_to_xpu()

from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

from torch.profiler import profile, record_function, ProfilerActivity
activities = [ProfilerActivity.CPU, ProfilerActivity.XPU]

# # paint
start_time = time.perf_counter()
max_num_view = 6  # can be 6 to 9
resolution = 512  # can be 768 or 512
conf = Hunyuan3DPaintConfig(max_num_view, resolution)
conf.realesrgan_ckpt_path = "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
conf.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
conf.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"
paint_pipeline = Hunyuan3DPaintPipeline(conf)

output_mesh_path = 'demo_textured.glb'
# with profile(activities=activities, record_shapes=False) as prof:
#     output_mesh_path = paint_pipeline(
#         mesh_path = "demo.glb", 
#         image_path = 'assets/demo.png',
#         output_mesh_path = output_mesh_path
#     )
# print(prof.key_averages().table(sort_by="self_xpu_time_total", row_limit=-1))
output_mesh_path = paint_pipeline(
    mesh_path = "demo.glb", 
    image_path = 'assets/demo.png',
    output_mesh_path = output_mesh_path
)
end_time = time.perf_counter()
print("Texture generation time: {:.2f} seconds".format(end_time - start_time))