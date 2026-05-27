import torch
from tqdm.notebook import tqdm
from .simclr_loss import nt_xent_loss
from .metrics import compute_classification_metrics


def ssl_train_one_epoch(model, loader, optimizer, device,
                        temperature=0.5, epoch=None, total_epochs=None):
    """Stage 1: SSL pretraining. No labels used."""
    model.train()
    running_loss = 0.0
    desc = f"SSL Epoch {epoch}/{total_epochs}" if epoch else "SSL Training"

    for view1, view2, _ in tqdm(loader, desc=desc, leave=False):
        view1, view2 = view1.to(device), view2.to(device)
        optimizer.zero_grad()
        _, z1 = model(view1)
        _, z2 = model(view2)
        loss = nt_xent_loss(z1, z2, temperature)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * view1.size(0)

    return running_loss / len(loader.dataset)


def classifier_train_one_epoch(backbone, classifier, loader, criterion,
                                optimizer, device, epoch=None, total_epochs=None):
    """Stage 2/3: supervised training on top of (frozen or unfrozen) backbone."""
    backbone.train()
    classifier.train()
    running_loss, all_preds, all_labels = 0.0, [], []
    desc = f"Cls Epoch {epoch}/{total_epochs}" if epoch else "Classifier Train"

    for images, labels in tqdm(loader, desc=desc, leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            h = backbone(images)
            logits = classifier(h)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        all_preds.extend(logits.argmax(1).detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return running_loss / len(loader.dataset), compute_classification_metrics(all_labels, all_preds)


@torch.no_grad()
def classifier_evaluate(backbone, classifier, loader, criterion, device,
                         epoch=None, total_epochs=None):
    """Evaluation for Stage 2/3."""
    backbone.eval()
    classifier.eval()
    running_loss, all_preds, all_labels = 0.0, [], []
    desc = f"Val Epoch {epoch}/{total_epochs}" if epoch else "Validation"

    for images, labels in tqdm(loader, desc=desc, leave=False):
        images, labels = images.to(device), labels.to(device)
        h = backbone(images)
        logits = classifier(h)
        loss = criterion(logits, labels)
        running_loss += loss.item() * images.size(0)
        all_preds.extend(logits.argmax(1).detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return running_loss / len(loader.dataset), compute_classification_metrics(all_labels, all_preds)