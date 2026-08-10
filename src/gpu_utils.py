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
