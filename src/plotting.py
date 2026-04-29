# src/plotting.py

import matplotlib.pyplot as plt


def plot_equity_curve(result, title, out_path):
    plt.figure(figsize=(10, 5))
    plt.plot(result["timestamp"], result["gross_equity"], label="Gross")
    plt.plot(result["timestamp"], result["net_equity"], label="Net")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_drawdown(result, title, out_path):
    equity = result["net_equity"]
    peak = equity.cummax()
    dd = equity / peak - 1

    plt.figure(figsize=(10, 5))
    plt.plot(result["timestamp"], dd)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
