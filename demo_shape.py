import sys
sys.path.insert(0, './hy3dshape')
sys.path.insert(0, './hy3dpaint')
import time
from PIL import Image
from hy3dshape.rembg import BackgroundRemover
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

import torch
from torch import nn

# import intel_extension_for_pytorch

import math
_original_layer_norm_forward = nn.LayerNorm.forward

def _new_layer_norm_forward(self, hidden_states: torch.Tensor):
    if (
        hidden_states.device.type == 'xpu' and 
        hidden_states.dtype in (torch.float, torch.half) and
        self.weight is not None
    ):
        try:
            import xe_addons
            hidden_size = math.prod(self.normalized_shape)
            x_2d = hidden_states.reshape(-1, hidden_size).contiguous()
            output = xe_addons.layer_norm(x_2d, self.weight, self.bias, self.eps)
            return output.reshape(hidden_states.shape)

            # import intel_extension_for_pytorch.ipex.llm.functional as ipex_test

            # hidden_states = ipex_test.fast_layer_norm(
            #     hidden_states,
            #     self.normalized_shape,
            #     self.weight,
            #     self.bias,
            #     self.eps,
            # )
            return hidden_states
        except ImportError:
            return _original_layer_norm_forward(self, hidden_states)
    else:
        print(hidden_states.dtype)
        return _original_layer_norm_forward(self, hidden_states)

nn.LayerNorm.forward = _new_layer_norm_forward

from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

try:
    from torchvision_fix import apply_fix
    apply_fix()
except ImportError:
    print("Warning: torchvision_fix module not found, proceeding without compatibility fix")                                      
except Exception as e:
    print(f"Warning: Failed to apply torchvision fix: {e}")

from torch.profiler import profile, record_function, ProfilerActivity
activities = [ProfilerActivity.CPU, ProfilerActivity.XPU]

# # shape
start_time = time.perf_counter()
model_path = '/llm/models/Hunyuan3D-2.1/'
pipeline_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)
image_path = 'assets/demo.png'
image = Image.open(image_path).convert("RGBA")
if image.mode == 'RGB':
    rembg = BackgroundRemover()
    image = rembg(image)
# image = ["/llm/workspace/chair_images_test/Picture1.png",
#         "/llm/workspace/chair_images_test/Picture2.png",
#         "/llm/workspace/chair_images_test/Picture3.png",]

# with profile(activities=activities, record_shapes=False) as prof:
mesh = pipeline_shapegen(image=image)[0]
mesh.export('demo.glb')
end_time = time.perf_counter()
print("Shape generation time: {:.2f} seconds".format(end_time - start_time))

# print(prof.key_averages().table(sort_by="self_xpu_time_total", row_limit=-1))
