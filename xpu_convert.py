import torch
from torch import nn
import torch.nn.functional as F
from diffusers.models.attention_processor import AttnProcessor2_0, Attention
from typing import Callable, List, Optional, Tuple, Union
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


def chunk_scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    chunk_size=1024
):
    """
    分块计算的缩放点积注意力，用于减少显存峰值使用
    
    参数:
        query, key, value: 输入张量
        attn_mask: 注意力掩码 (支持None或与query/key形状兼容的掩码)
        dropout_p: dropout概率
        is_causal: 是否使用因果注意力
        chunk_size: 分块大小 (根据可用显存调整)
    
    返回:
        注意力计算结果张量
    """
    # 如果不分块或query序列长度小于分块大小，直接调用原始函数
    if chunk_size is None or query.size(2) <= chunk_size:
        return F.scaled_dot_product_attention(
            query, key, value, attn_mask, dropout_p, is_causal
        )
    
    # 检查是否支持分块的情况
    if is_causal:
        warnings.warn("Chunked computation may not work correctly with causal attention. "
                      "Consider setting chunk_size=None for causal attention.")
    
    if dropout_p > 0:
        warnings.warn("Dropout is applied independently to each chunk, which may "
                      "result in slightly different behavior compared to non-chunked version.")
    
    # 获取query序列长度
    Lq = query.size(2)
    
    # 分块处理query
    query_chunks = torch.split(query, chunk_size, dim=2)
    
    # 处理注意力掩码的分块
    mask_chunks = None
    if attn_mask is not None:
        # 确定掩码的分块维度
        split_dim = -2 if attn_mask.dim() >= 2 else 0
        
        # 如果掩码在query维度上可广播，则每个块使用相同掩码
        if attn_mask.size(split_dim) == 1:
            mask_chunks = [attn_mask] * len(query_chunks)
        # 否则将掩码按query维度分块
        elif attn_mask.size(split_dim) == Lq:
            mask_chunks = torch.split(attn_mask, chunk_size, dim=split_dim)
        else:
            raise ValueError(f"Attention mask size {attn_mask.size()} is incompatible "
                             f"with query size {query.size()} for chunked computation")
    else:
        mask_chunks = [None] * len(query_chunks)
    
    # 分块计算结果存储
    output_chunks = []
    
    # 逐块计算注意力
    for q_chunk, m_chunk in zip(query_chunks, mask_chunks):
        chunk_output = F.scaled_dot_product_attention(
            q_chunk, key, value, 
            attn_mask=m_chunk,
            dropout_p=dropout_p,
            is_causal=is_causal
        )
        output_chunks.append(chunk_output)
    
    # 合并所有块的结果
    return torch.cat(output_chunks, dim=2)

def chunked_diffusers_attention_processor_call(
    self,
    attn: Attention,
    hidden_states: torch.Tensor,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    temb: Optional[torch.Tensor] = None,
    *args,
    **kwargs,
) -> torch.Tensor:
    if len(args) > 0 or kwargs.get("scale", None) is not None:
        deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
        deprecate("scale", "1.0.0", deprecation_message)

    residual = hidden_states
    if attn.spatial_norm is not None:
        hidden_states = attn.spatial_norm(hidden_states, temb)

    input_ndim = hidden_states.ndim

    if input_ndim == 4:
        batch_size, channel, height, width = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

    batch_size, sequence_length, _ = (
        hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
    )

    if attention_mask is not None:
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        # scaled_dot_product_attention expects attention_mask shape to be
        # (batch, heads, source_length, target_length)
        attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

    if attn.group_norm is not None:
        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

    query = attn.to_q(hidden_states)

    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states
    elif attn.norm_cross:
        encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

    key = attn.to_k(encoder_hidden_states)
    value = attn.to_v(encoder_hidden_states)

    inner_dim = key.shape[-1]
    head_dim = inner_dim // attn.heads

    query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    if attn.norm_q is not None:
        query = attn.norm_q(query)
    if attn.norm_k is not None:
        key = attn.norm_k(key)

    # the output of sdp = (batch, num_heads, seq_len, head_dim)
    # TODO: add support for attn.scale when we move to Torch 2.1
    hidden_states = chunk_scaled_dot_product_attention(
        query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
    )

    hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
    hidden_states = hidden_states.to(query.dtype)

    # linear proj
    hidden_states = attn.to_out[0](hidden_states)
    # dropout
    hidden_states = attn.to_out[1](hidden_states)

    if input_ndim == 4:
        hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

    if attn.residual_connection:
        hidden_states = hidden_states + residual

    hidden_states = hidden_states / attn.rescale_output_factor

    return hidden_states

from realesrgan import RealESRGANer
def process_on_xpu(self):
    self.output = self.model(self.img.to("xpu"))

def convert_to_xpu():
    nn.LayerNorm.forward = _new_layer_norm_forward
    AttnProcessor2_0.__call__ = chunked_diffusers_attention_processor_call
    RealESRGANer.process = process_on_xpu

    print("Converted to XPU compatible functions.")