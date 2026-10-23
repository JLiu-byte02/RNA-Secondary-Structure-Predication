"""Explicit numerical runtime; CPU scores are recomputed for every role."""
def execution_spec(device):
    if device not in ('cpu','cuda'):raise ValueError('Explicit cpu/cuda runtime required')
    return dict(device=device,threads=2,precision='FP32' if device=='cpu' else 'BF16_autocast',
                base_logits='recompute_fp32_all_roles' if device=='cpu' else 'historical_BF16_cache_and_matching_CUDA_inference')
