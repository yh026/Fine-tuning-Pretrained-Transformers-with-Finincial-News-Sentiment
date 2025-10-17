"""
Utility: functions for visualization
Author: Chen Yuhan
Last Edited: 2025.10.17
"""

from matplotlib import pyplot as plt
import pandas as pd
import os
def plot_train_loss(full_logs, lora_logs, save_dir):
    """
    Plot the training loss.
    """
    logs = pd.concat([full_logs, lora_logs])
    # Filter only training logs (where eval_loss is NaN)
    train_logs = logs[~logs["loss"].isna()]

    plt.figure(figsize=(8, 5))
    for setup, df in train_logs.groupby("setup"):
        plt.plot(df["epoch"], df["loss"], label=setup)

    plt.title("Training Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(save_dir, "train_loss_comparison.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_eval_loss(full_logs,lora_logs,save_dir):
    """
    Plot the evaluation loss.
    """
    eval_full = full_logs[full_logs["eval_loss"].notna()][["epoch", "eval_loss", "eval_accuracy", "eval_f1"]]
    eval_lora = lora_logs[lora_logs["eval_loss"].notna()][["epoch", "eval_loss", "eval_accuracy", "eval_f1"]]
    plt.figure(figsize=(8,5))
    plt.plot(eval_full["epoch"][0:20], eval_full["eval_loss"][0:20], label="Full Fine-Tuning", marker="o")
    plt.plot(eval_lora["epoch"], eval_lora["eval_loss"], label="LoRA Fine-Tuning", marker="o")

    # Find the minimum loss for best performance epoch
    min_idx_full = eval_full["eval_loss"][0:20].idxmin()
    min_epoch_full = eval_full.loc[min_idx_full, "epoch"]
    plt.axvline(x=min_epoch_full, color='blue', linestyle='--', alpha=0.7, label=f'Full Min (Epoch {min_epoch_full:.0f})')

    min_idx_lora = eval_lora["eval_loss"].idxmin()
    min_epoch_lora = eval_lora.loc[min_idx_lora, "epoch"]
    plt.axvline(x=min_epoch_lora, color='orange', linestyle='--', alpha=0.7, label=f'LoRA Min (Epoch {min_epoch_lora:.0f})')

    plt.title("Validation Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Eval Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(save_dir, "validation_loss_comparison.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_accuracy_f1(full_logs,lora_logs,save_dir):
    logs = pd.concat([full_logs, lora_logs])
    # Keep only eval metrics
    eval_logs = logs[logs["loss"].isna()]  # only evaluation steps

    # Plot Accuracy & F1
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for setup, df in eval_logs.groupby("setup"):
        ax[0].plot(df["epoch"], df["eval_accuracy"], label=setup)
        ax[1].plot(df["epoch"], df["eval_f1"], label=setup)

        epoch_6_data = df[df["epoch"] == 6]

        acc_val = epoch_6_data["eval_accuracy"].values[0]
        f1_val = epoch_6_data["eval_f1"].values[0]

        ax[0].annotate(f'{acc_val:.3f}',
                    xy=(6, acc_val),
                    xytext=(10, 10),
                    textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

        ax[1].annotate(f'{f1_val:.3f}',
                    xy=(6, f1_val),
                    xytext=(10, 10),
                    textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))


    ax[0].set_title("Validation Accuracy per Epoch")
    ax[1].set_title("Validation F1 per Epoch")
    for a in ax:
        a.set_xlabel("Epoch")
        a.set_ylabel("Score")
        a.legend()
        a.grid(True)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, "accuracy_f1_comparison.png")
    plt.savefig(save_path, dpi=300)
    plt.close()


def create_comparison_plots(full_logs, lora_logs):
    save_dir = "./results"
    os.makedirs(save_dir, exist_ok=True)
    plot_train_loss(full_logs, lora_logs,save_dir)
    plot_eval_loss(full_logs, lora_logs,save_dir)
    plot_accuracy_f1(full_logs, lora_logs,save_dir)
