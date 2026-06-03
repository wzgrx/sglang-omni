"""Blackwell (SM120) compatibility patch for flash-attention.

flash-attn-4's precompiled .so has SM120 backward kernels but not forward kernels.
This patch monkey-patches flash_attn to use PyTorch SDPA as fallback on SM120.
"""
from __future__ import annotations

import torch
import logging

logger = logging.getLogger(__name__)

_original_fwd = None
_original_varlen_fwd = None


def is_blackwell() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(0)
    return major >= 12


def _sdp_fallback_fwd(q, k, v, *args, **kwargs):
    """Use PyTorch scaled_dot_product_attention as fallback."""
    import torch.nn.functional as F
    softmax_scale = kwargs.get("softmax_scale", None)
    causal = kwargs.get("causal", False)
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    attn = F.scaled_dot_product_attention(
        q, k, v, attn_mask=None, dropout_p=0.0,
        is_causal=causal, scale=softmax_scale,
    )
    return attn, None, None, None


def _sdp_fallback_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k,
                          max_seqlen_q, max_seqlen_k, *args, **kwargs):
    """Variable-length SDPA fallback."""
    import torch.nn.functional as F
    softmax_scale = kwargs.get("softmax_scale", None)
    causal = kwargs.get("causal", False)
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    outputs = []
    for i in range(len(cu_seqlens_q) - 1):
        q_i = q[cu_seqlens_q[i]:cu_seqlens_q[i + 1]].unsqueeze(0)
        k_i = k[cu_seqlens_k[i]:cu_seqlens_k[i + 1]].unsqueeze(0)
        v_i = v[cu_seqlens_k[i]:cu_seqlens_k[i + 1]].unsqueeze(0)
        attn = F.scaled_dot_product_attention(
            q_i, k_i, v_i, attn_mask=None, dropout_p=0.0,
            is_causal=causal, scale=softmax_scale,
        )
        outputs.append(attn.squeeze(0))
    out = torch.cat(outputs, dim=0)
    return out, None, None, None


def apply_patch():
    """Apply Blackwell fallback patches to flash_attn module."""
    global _original_fwd, _original_varlen_fwd
    if not is_blackwell():
        return False
    try:
        import flash_attn.flash_attn_interface as fi
        _original_fwd = fi._flash_attn_forward
        _original_varlen_fwd = fi._flash_attn_varlen_forward
        fi._flash_attn_forward = _sdp_fallback_fwd
        fi._flash_attn_varlen_forward = _sdp_fallback_varlen
        logger.info("Applied flash_attn Blackwell SM120 SDPA fallback patch")
        return True
    except Exception as e:
        logger.warning(f"Could not apply flash_attn patch: {e}")
        return False


def remove_patch():
    global _original_fwd, _original_varlen_fwd
    if _original_fwd is None:
        return
    try:
        import flash_attn.flash_attn_interface as fi
        fi._flash_attn_forward = _original_fwd
        fi._flash_attn_varlen_forward = _original_varlen_fwd
    except Exception:
        pass
