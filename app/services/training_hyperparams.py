def hf_optimizer_name(name):
    return name

def yolo_optimizer_name(name):
    return name

def torch_optimizer_class(name):
    import torch
    return torch.optim.Adam

def torch_optimizer_kwargs(name):
    return {}
