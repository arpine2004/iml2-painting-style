import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from IPython.display import clear_output, display


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

from IPython.display import clear_output
import matplotlib.pyplot as plt

def plot_training_history_live(history, epoch=None, total_epochs=None):
    clear_output(wait=True)

    epochs = range(1, len(history["train_loss"]) + 1)
    title = "Training History"
    if epoch is not None and total_epochs is not None:
        title = f"Training History - Epoch {epoch}/{total_epochs}"

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss")
    axes[0].set_title(title + " | Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_accuracy"], label="Train Accuracy")
    axes[1].plot(epochs, history["val_accuracy"], label="Val Accuracy")
    axes[1].set_title(title + " | Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    axes[2].plot(epochs, history["train_macro_f1"], label="Train Macro F1")
    axes[2].plot(epochs, history["val_macro_f1"], label="Val Macro F1")
    axes[2].set_title(title + " | Macro F1")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Macro F1")
    axes[2].legend()

    plt.tight_layout()
    plt.show()