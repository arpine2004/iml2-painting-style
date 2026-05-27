import torch
import torch.nn.functional as F


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """
    Computes the NT-Xent (Normalized Temperature-scaled Cross Entropy) loss for a batch of paired representations (z1, z2).
    z1 and z2 are the outputs of the encoder for two augmented views of the same images.
    """
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)         

    sim = torch.mm(z, z.T) / temperature  

    mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, float('-inf'))

    labels = torch.cat([
        torch.arange(B, 2 * B, device=z.device),
        torch.arange(0, B, device=z.device),
    ])

    return F.cross_entropy(sim, labels)