import pickle
import torch
import numpy as np
from glob import glob
import os

import os
import pickle
import torch
import numpy as np
from glob import glob

base_dir = "save/pems08_Tl_50"
pk_pattern = "**/generated_outputs_nsample100.pk"

pk_files = glob(os.path.join(base_dir, pk_pattern), recursive=True)
print(f"Found {len(pk_files)} runs")

all_samples = []
all_targets = []
all_evalpoints = []
all_observed_points = []
all_observed_times = []

scaler_ref = None
mean_scaler_ref = None

for pk_file in pk_files:
    with open(pk_file, "rb") as f:
        samples, target, evalpoint, obs_point, obs_time, scaler, mean_scaler = pickle.load(f)

    # consistency checks
    if scaler_ref is None:
        scaler_ref = scaler
        mean_scaler_ref = mean_scaler
    else:
        assert torch.allclose(scaler, scaler_ref), f"Scaler mismatch in {pk_file}"
        assert torch.allclose(mean_scaler, mean_scaler_ref), f"Mean scaler mismatch in {pk_file}"

    all_samples.append(samples)
    all_targets.append(target)
    all_evalpoints.append(evalpoint)
    all_observed_points.append(obs_point)
    all_observed_times.append(obs_time)

# concatenate over batch dimension
all_generated_samples = torch.cat(all_samples, dim=0)
all_target = torch.cat(all_targets, dim=0)
all_evalpoint = torch.cat(all_evalpoints, dim=0)
all_observed_point = torch.cat(all_observed_points, dim=0)
all_observed_time = torch.cat(all_observed_times, dim=0)

print("Combined shape:", all_generated_samples.shape)

with open("save/pems08_Tl_50/combined_outputs_nsample100.pk", "wb") as f:
    pickle.dump(
        [
            all_generated_samples,
            all_target,
            all_evalpoint,
            all_observed_point,
            all_observed_time,
            scaler_ref,
            mean_scaler_ref,
        ],
        f,
    )

import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 11,
    "figure.dpi": 150
})

# convert to numpy
samples = all_generated_samples.cpu().numpy()
target = all_target.cpu().numpy()
evalpoint = all_evalpoint.cpu().numpy()

scaler = scaler_ref.cpu().numpy()
mean_scaler = mean_scaler_ref.cpu().numpy()

# choose sensor k and batch index b to plot
b, k = 0, 0

samp = np.transpose(samples[b], (1, 0, 2))  # (L, nsample, K)
median = np.median(samp, axis=1)
q05 = np.quantile(samp, 0.05, axis=1)
q95 = np.quantile(samp, 0.95, axis=1)

# rescale
tgt = target[b] * scaler + mean_scaler
median = median * scaler + mean_scaler
q05 = q05 * scaler + mean_scaler
q95 = q95 * scaler + mean_scaler

plt.figure(figsize=(10, 4))
plt.plot(tgt[:, k], color="black", linewidth=1.5, label="Ground Truth")
plt.plot(median[:, k], color="tab:blue", linewidth=2, label="Median Prediction")
plt.fill_between(
    range(len(tgt)),
    q05[:, k],
    q95[:, k],
    color="tab:blue",
    alpha=0.25,
    label="90% Prediction Interval",
)

plt.xlabel("Time")
plt.ylabel("Traffic Flow")
plt.title(f"PEMS08 Sensor {k+1}")
plt.legend(frameon=False)
plt.tight_layout()
plt.show()
