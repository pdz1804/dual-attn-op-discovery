# utils/plot_utils.py

import matplotlib.pyplot as plt
import os

def _prepare_output_dir():
    """
    Ensure that 'data/outputs/img/' exists.
    """
    output_dir = os.path.join("data", "outputs", "img")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def plot_loss(training_history, title="Training Loss", save_path=None):
    """
    Plot training and validation loss over epochs and save to 'data/outputs/img/'.

    Args:
        training_history (dict): Dictionary containing 'train_loss' and 'val_loss'.
        title (str): Title of the plot.
        save_path (str): Optional path to save the figure. If None, saves to 'data/outputs/img/loss.png'.
    """
    output_dir = _prepare_output_dir()
    if save_path is None:
        save_path = os.path.join(output_dir, "loss.png")

    plt.figure(figsize=(8,5))
    plt.plot(training_history["train_loss"], label="Train Loss")
    plt.plot(training_history["val_loss"], label="Val Loss")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"[SAVE] Loss plot saved to: {save_path}")

def plot_accuracy(training_history, title="Training Accuracy", save_path=None):
    """
    Plot training and validation accuracy over epochs and save to 'data/outputs/img/'.

    Args:
        training_history (dict): Dictionary containing 'train_metrics' and 'val_metrics'.
        title (str): Title of the plot.
        save_path (str): Optional path to save the figure. If None, saves to 'data/outputs/img/accuracy.png'.
    """
    output_dir = _prepare_output_dir()
    if save_path is None:
        save_path = os.path.join(output_dir, "accuracy.png")

    train_acc = [m["accuracy"] for m in training_history["train_metrics"]]
    val_acc = [m["accuracy"] for m in training_history["val_metrics"]]

    plt.figure(figsize=(8,5))
    plt.plot(train_acc, label="Train Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"[SAVE] Accuracy plot saved to: {save_path}")



