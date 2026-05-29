import numpy as np
import torch
from tqdm.notebook import tqdm
from .vae import vae_loss


def vae_train_one_epoch(model, loader, optimizer, device, beta=1.0, epoch=None, total_epochs=None):
    model.train()
    total, recon_t, kl_t = 0.0, 0.0, 0.0
    desc = f"VAE Epoch {epoch}/{total_epochs}" if epoch else "VAE Train"

    for images, _ in tqdm(loader, desc=desc, leave=False):
        images = images.to(device)
        optimizer.zero_grad()
        recon, mu, logvar = model(images)
        loss, rl, kl = vae_loss(recon, images, mu, logvar, beta)
        loss.backward()
        optimizer.step()
        n = images.size(0)
        total += loss.item() * n
        recon_t += rl.item() * n
        kl_t += kl.item() * n

    N = len(loader.dataset)
    return total / N, recon_t / N, kl_t / N


@torch.no_grad()
def vae_evaluate(model, loader, device, beta=1.0, epoch=None, total_epochs=None):
    model.eval()
    total, recon_t, kl_t = 0.0, 0.0, 0.0
    desc = f"VAE Val {epoch}/{total_epochs}" if epoch else "VAE Val"

    for images, _ in tqdm(loader, desc=desc, leave=False):
        images = images.to(device)
        recon, mu, logvar = model(images)
        loss, rl, kl = vae_loss(recon, images, mu, logvar, beta)
        n = images.size(0)
        total += loss.item() * n
        recon_t += rl.item() * n
        kl_t += kl.item() * n

    N = len(loader.dataset)
    return total / N, recon_t / N, kl_t / N


@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.eval()
    mus, labels = [], []
    for images, lbls in loader:
        mu, _ = model.encode(images.to(device))
        mus.append(mu.cpu())
        labels.extend(lbls.numpy())
    return torch.cat(mus).numpy(), np.array(labels)
