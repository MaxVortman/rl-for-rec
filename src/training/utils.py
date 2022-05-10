import torch
import random
import numpy as np


def t2d(tensor, device):
    """Move tensors to a specified device.

    Args:
        tensor (torch.Tensor or Dict[str, torch.Tensor] or list/tuple of torch.Tensor):
            data to move to a device.
        device (str or torch.device): device where should be moved device

    Returns:
        torch.Tensor or Dict[str, torch.Tensor] or List[torch.Tensor] based on `tensor` type.
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.to(device)
    elif isinstance(tensor, (tuple, list)):
        # recursive move to device
        return [t2d(_tensor, device) for _tensor in tensor]
    elif isinstance(tensor, dict):
        res = {}
        for _key, _tensor in tensor.items():
            res[_key] = t2d(_tensor, device)
        return res


def seed_all(seed, deterministic=True, benchmark=True) -> None:
    """Fix all seeds so results can be reproducible.

    Args:
        seed (int): random seed.
            Default is `42`.
        deterministic (bool): flag to use cuda deterministic
            algoritms for computations.
            Default is `True`.
        benchmark (bool): flag to use benchmark option
            to select the best algorithm for computatins.
            Should be used `True` with fixed size
            data (images or similar) for other types of
            data is better to use `False` option.
            Default is `True`.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    random.seed(seed)
    np.random.seed(seed)
    # reproducibility
    torch.backends.cudnn.deterministic = deterministic
    # small speedup
    torch.backends.cudnn.benchmark = benchmark


def log_metrics(metrics, loader):
    """Write metrics to tensorboard and stdout.

    Args:
        metrics (dict): metrics computed during training/validation steps
        loader (str): loader name
    """
    output_line = []
    for name in metrics:
        value = metrics[name]
        output_line.append(f"{name} - {value:.5f}")
    output_line = f"{loader}: " + ", ".join(output_line)
    print(output_line)


def soft_update(target_net, net, soft_tau=1e-2):
    for target_param, param in zip(target_net.parameters(), net.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - soft_tau) + param.data * soft_tau
        )
