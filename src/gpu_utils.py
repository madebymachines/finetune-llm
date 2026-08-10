import gc

import torch


def check_cuda():
    available = torch.cuda.is_available()
    info = {"available": available}
    if available:
        props = torch.cuda.get_device_properties(0)
        info["name"] = props.name
        info["total_gb"] = round(props.total_memory / 1024**3, 2)
        info["reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
    return info


def memory_snapshot():
    if not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_reserved() / 1024**3, 3)


def clear_gpu_cache():
    """Release whatever CUDA memory is reclaimable (freed training buffers,
    stale optimizer scratch space, fragmented cache blocks) without touching
    the model/adapter weights themselves. Doesn't reduce memory the model
    itself needs — just gives back memory PyTorch is holding onto but not
    actively using, so callers should still watch the reserved_gb reading
    after calling this rather than assume it fully fixes an OOM."""
    if not torch.cuda.is_available():
        return
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
