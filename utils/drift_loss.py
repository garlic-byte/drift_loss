import torch
import torch.nn.functional as F


def compute_drift(gen: torch.Tensor, pos: torch.Tensor, temp: float = 0.05) -> torch.Tensor:
    """
    Compute drift field V with attention-based kernel.

    Args:
        gen: Generated samples [G, D]
        pos: Data samples [P, D]
        temp: Temperature for softmax kernel

    Returns:
        V: Drift vectors [G, D]
    """
    targets = torch.cat([gen, pos], dim=0)
    G = gen.shape[0]

    dist = torch.cdist(gen, targets)
    dist[:, :G].fill_diagonal_(1e6)  # mask self
    kernel = (-dist / temp).exp()  # unnormalized kernel

    normalizer = kernel.sum(dim=-1, keepdim=True) * kernel.sum(dim=-2,
                                                               keepdim=True)  # normalize along both dimensions, which we found to slightly improve performance
    normalizer = normalizer.clamp_min(1e-12).sqrt()
    normalized_kernel = kernel / normalizer

    pos_coeff = normalized_kernel[:, G:] * normalized_kernel[:, :G].sum(dim=-1, keepdim=True)
    pos_V = pos_coeff @ targets[G:]
    neg_coeff = normalized_kernel[:, :G] * normalized_kernel[:, G:].sum(dim=-1, keepdim=True)
    neg_V = neg_coeff @ targets[:G]

    return pos_V - neg_V

def drifting_loss(gen: torch.Tensor, pos: torch.Tensor):
    """Drifting loss: MSE(gen, stopgrad(gen + V))."""

    with torch.no_grad():
        V = compute_drift(gen, pos, 100)
        target = (gen + V).detach()
    loss = F.mse_loss(gen, target)
    return loss

def get_drift_loss(inputs, outputs, labels, num_classes):
    """
    Calculate drift loss for each class of datasets.
    :param inputs: (batch, channels, height, width)
    :param outputs: (batch, channels, height, width)
    :param labels: (batch_size)
    :param num_classes: int
    :return: float
    """
    total_loss = torch.tensor(0.0, device=inputs.device, requires_grad=True)
    for class_idx in range(num_classes):
        idx_indices = torch.where(labels==class_idx)[0]
        class_inputs = inputs[idx_indices].flatten(1) # [batch_size, hidden_dim]
        class_outputs = outputs[idx_indices].flatten(1) # [batch_size, hidden_dim]

        loss = drifting_loss(class_outputs, class_inputs)
        total_loss = total_loss + loss

    return total_loss / num_classes