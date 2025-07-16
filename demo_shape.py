import sys
sys.path.insert(0, './hy3dshape')
sys.path.insert(0, './hy3dpaint')
import time
from PIL import Image
from hy3dshape.rembg import BackgroundRemover
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

import torch
from torch import nn

from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

try:
    from torchvision_fix import apply_fix
    apply_fix()
except ImportError:
    print("Warning: torchvision_fix module not found, proceeding without compatibility fix")                                      
except Exception as e:
    print(f"Warning: Failed to apply torchvision fix: {e}")

from xpu_convert import convert_to_xpu
convert_to_xpu()

# # shape
start_time = time.perf_counter()
model_path = 'tencent/Hunyuan3D-2.1'
pipeline_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)
image_path = 'assets/demo.png'
image = Image.open(image_path).convert("RGBA")
if image.mode == 'RGB':
    rembg = BackgroundRemover()
    image = rembg(image)

mesh = pipeline_shapegen(image=image)[0]
mesh.export('demo.glb')
end_time = time.perf_counter()
print("Shape generation time: {:.2f} seconds".format(end_time - start_time))
