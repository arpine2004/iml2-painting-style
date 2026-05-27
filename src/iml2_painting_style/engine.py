import torch
from tqdm.notebook import tqdm
from .metrics import compute_classification_metrics


def train_one_epoch(model, loader, criterion, optimizer, device, epoch=None, total_epochs=None):
    model.train()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    desc = f"Train Epoch {epoch}/{total_epochs}" if epoch is not None and total_epochs is not None else "Training"

    progress_bar = tqdm(loader, desc=desc, leave=False)

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_classification_metrics(all_labels, all_preds)

    return epoch_loss, metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device, epoch=None, total_epochs=None):
    model.eval()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    desc = f"Val Epoch {epoch}/{total_epochs}" if epoch is not None and total_epochs is not None else "Validation"

    progress_bar = tqdm(loader, desc=desc, leave=False)

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_classification_metrics(all_labels, all_preds)

    return epoch_loss, metrics