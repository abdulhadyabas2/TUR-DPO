from __future__ import annotations
import matplotlib.pyplot as plt


def plot_curve(xs, ys, xlabel, ylabel, title, out_path):
    plt.figure(figsize=(5, 3))
    plt.plot(xs, ys, marker='o')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
