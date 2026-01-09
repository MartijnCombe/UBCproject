import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

exp_files = {
    "Run 1": "save/pems08_full_Training/combined_outputs_nsample100.pk",
    "Run 2": "save/pems08_Tl_50/combined_outputs_nsample100.pk",
    "Run 3": "save/pems08_Tl_75/combined_outputs_nsample100.pk",
}

data = {}

for label, path in exp_files.items():
    with open(path, "rb") as f:
        samples, target, evalpoint, obs_point, obs_time, scaler, mean_scaler = pickle.load(f)

    data[label] = {
        "samples": samples.cpu().numpy(),
        "target": target.cpu().numpy(),
        "evalpoint": evalpoint.cpu().numpy(),
        "scaler": scaler.cpu().numpy(),
        "mean_scaler": mean_scaler.cpu().numpy(),
    }

b = 50  # same batch index
k = 40  # same sensor index

T = data["Run 1"]["target"].shape[1]  # time length

plt.figure(figsize=(11, 4))

# Ground truth (same for all)
tgt = data["Run 1"]["target"][b] * data["Run 1"]["scaler"] + data["Run 1"]["mean_scaler"]
plt.plot(
    tgt[:, k],
    color="black",
    linewidth=2,
    label="Ground Truth"
)

colors = ["tab:blue", "tab:orange", "tab:green"]

new_labels = {
    "Run 1": "PriSTI",
    "Run 2": "PriSTI_TL_50",
    "Run 3": "PriSTI_TL_75",
}

for (label, d), color in zip(data.items(), colors):
    samp = np.transpose(d["samples"][b], (1, 0, 2))  # (L, nsample, K)

    median = np.median(samp, axis=1)
    median = median * d["scaler"] + d["mean_scaler"]

    plt.plot(
        median[:, k],
        color=color,
        linewidth=1.8,
        label=new_labels[label]  # <-- use new label here
    )
    
figures_folder = "figure"
os.makedirs(figures_folder, exist_ok=True)
filename = f"sensor_{k+1}_comparison.png"
filepath = os.path.join(figures_folder, filename)

plt.xlabel("Time Step")
plt.ylabel("Value")
plt.title(f"PEMS08 Sensor {k+1} – Comparison Across Models")
plt.legend(frameon=False, ncol=4)
plt.tight_layout()

# Save the figure
plt.savefig(filepath, dpi=300, bbox_inches="tight")
plt.show()
