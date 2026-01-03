import os
import pickle
import numpy as np
import torch
import torchcde
from torch.utils.data import DataLoader, Dataset

from utils import get_randmask, get_block_mask

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def _load_solar_npz(npz_path: str):
    arr = np.load(npz_path, allow_pickle=True)
    data = arr["data"]  # (T, N, 1)
    # use first feature -> (T, N)
    data = data[..., 0]
    timestamps = arr.get("timestamps", None)  # (T,)
    node_ids = arr.get("node_ids", None)      # (N,)
    latlon = arr.get("latlon", None)          # (N, 2)
    return data, timestamps, node_ids, latlon


def get_solar_mean_std(npz_path: str, out_path: str, train_ratio: float = 0.7):
    data, _, _, _ = _load_solar_npz(npz_path)  # (T, N)

    # observed mask: treat non-nan as observed; 
    ob_mask = np.isfinite(data).astype("uint8")
   
    T = data.shape[0]
    train_T = int(T * train_ratio)

    train_data = np.nan_to_num(data[:train_T], nan=0.0)
    train_mask = ob_mask[:train_T]

    # mean/std per node using only observed entries
    denom = train_mask.sum(axis=0).clip(min=1)
    mean = (train_data * train_mask).sum(axis=0) / denom

    var = ((train_data - mean) * train_mask) ** 2
    std = np.sqrt(var.sum(axis=0) / denom)
    std = np.where(std == 0, 1.0, std)

    with open(out_path, "wb") as f:
        pickle.dump((mean.astype(np.float32), std.astype(np.float32)), f)


def sample_mask(shape, p=0.0015, p_noise=0.05, max_seq=1, min_seq=1, rng=None):
    
    if rng is None:
        rand = np.random.random
        randint = np.random.randint
    else:
        rand = rng.random
        randint = rng.integers
    mask = rand(shape) < p
    for col in range(mask.shape[1]):
        idxs = np.flatnonzero(mask[:, col])
        if not len(idxs):
            continue
        fault_len = min_seq
        if max_seq > min_seq:
            fault_len = fault_len + int(randint(max_seq - min_seq))
        idxs_ext = np.concatenate([np.arange(i, i + fault_len) for i in idxs])
        idxs = np.unique(idxs_ext)
        idxs = np.clip(idxs, 0, shape[0] - 1)
        mask[idxs, col] = True
    mask = mask | (rand(mask.shape) < p_noise)
    return mask.astype("uint8")


class Solar_Dataset(Dataset):
    def __init__(
        self,
        npz_path: str = "./processed_for_pristi_repo.npz",
        eval_length: int = 24,
        mode: str = "train",
        val_len: float = 0.1,
        test_len: float = 0.2,
        missing_pattern: str = "block",
        is_interpolate: bool = False,
        target_strategy: str = "random",
        meanstd_path: str = "data/solar/solar_meanstd.pk",
        seed: int = 9101112,
    ):
        self.eval_length = eval_length
        self.is_interpolate = is_interpolate
        self.target_strategy = target_strategy
        self.mode = mode
        self.npz_path = npz_path

        if meanstd_path is None:
            meanstd_path = os.path.splitext(npz_path)[0] + "_meanstd.pk"
        self.meanstd_path = meanstd_path

        if not os.path.exists(self.meanstd_path):
            get_solar_mean_std(npz_path=self.npz_path, out_path=self.meanstd_path, train_ratio=0.7)

        with open(self.meanstd_path, "rb") as f:
            self.train_mean, self.train_std = pickle.load(f)  # (N,), (N,)

        # Load data
        data, timestamps, node_ids, latlon = _load_solar_npz(self.npz_path)  # (T,N)
        self.timestamps = timestamps
        self.node_ids = node_ids
        self.latlon = latlon

        # observed mask: finite values are observed
        ob_mask = np.isfinite(data).astype("uint8")

        # create artificial eval mask
        self.rng = np.random.default_rng(seed)
        if missing_pattern == "block":
            eval_mask = sample_mask(shape=data.shape, p=0.0015, p_noise=0.05, min_seq=12, max_seq=12 * 4, rng=self.rng)
        elif missing_pattern == "point":
            eval_mask = sample_mask(shape=data.shape, p=0.0, p_noise=0.25, min_seq=12, max_seq=12 * 4, rng=self.rng)
        else:
            raise ValueError("missing_pattern must be 'block' or 'point'")

        # Ground-truth mask: observed but not masked-out by eval_mask
        gt_mask = (1 - (eval_mask | (1 - ob_mask))).astype("uint8")

        # Normalize using train mean/std and keep zeros at missing points
        c_data = ((np.nan_to_num(data, nan=0.0) - self.train_mean) / self.train_std) * ob_mask

        # Split
        val_start = int((1 - val_len - test_len) * len(data))
        test_start = int((1 - test_len) * len(data))

        if mode == "train":
            self.observed_mask = ob_mask[:val_start]
            self.gt_mask = gt_mask[:val_start]
            self.observed_data = c_data[:val_start]
        elif mode == "valid":
            self.observed_mask = ob_mask[val_start:test_start]
            self.gt_mask = gt_mask[val_start:test_start]
            self.observed_data = c_data[val_start:test_start]
        elif mode == "test":
            self.observed_mask = ob_mask[test_start:]
            self.gt_mask = gt_mask[test_start:]
            self.observed_data = c_data[test_start:]
        else:
            raise ValueError("mode must be 'train', 'valid', or 'test'")

        # build indices for sliding windows
        self.use_index = []
        self.cut_length = []

        current_length = len(self.observed_mask) - eval_length + 1
        if current_length <= 0:
            raise ValueError(f"eval_length={eval_length} is larger than split length={len(self.observed_mask)}")

        if mode == "test":
            n_sample = len(self.observed_data) // eval_length
            c_index = np.arange(0, 0 + eval_length * n_sample, eval_length)
            self.use_index += c_index.tolist()
            self.cut_length += [0] * len(c_index)
            if len(self.observed_data) % eval_length != 0:
                self.use_index += [current_length - 1]
                self.cut_length += [eval_length - len(self.observed_data) % eval_length]
        else:
            self.use_index = np.arange(current_length)
            self.cut_length = [0] * len(self.use_index)

    def __getitem__(self, org_index: int):
        index = self.use_index[org_index]
        ob_data = self.observed_data[index: index + self.eval_length]     # (L,N)
        ob_mask = self.observed_mask[index: index + self.eval_length]     # (L,N)
        gt_mask = self.gt_mask[index: index + self.eval_length]           # (L,N)

        ob_mask_t = torch.tensor(ob_mask).float()

        if self.mode != "train":
            cond_mask = torch.tensor(gt_mask).to(torch.float32)
        else:
            if self.target_strategy != "random":
                cond_mask = get_block_mask(ob_mask_t, target_strategy=self.target_strategy)
            else:
                cond_mask = get_randmask(ob_mask_t)

        s = {
            "observed_data": ob_data,                         # numpy (L,N)
            "observed_mask": ob_mask,                         # numpy uint8 (L,N)
            "gt_mask": gt_mask,                               # numpy uint8 (L,N)
            "timepoints": np.arange(self.eval_length),        # numpy (L,)
            "cut_length": self.cut_length[org_index],         # int
            "cond_mask": cond_mask,                           # torch (L,N)
        }

        if self.is_interpolate:
            tmp_data = torch.tensor(ob_data).to(torch.float64)  # (L,N)
            itp_data = torch.where(cond_mask == 0, float("nan"), tmp_data).to(torch.float32)

            # torchcde expects (batch, length, channels). Here "batch" == N nodes, channels==1
            coeffs = torchcde.linear_interpolation_coeffs(
                itp_data.permute(1, 0).unsqueeze(-1)  # (N,L,1)
            ).squeeze(-1).permute(1, 0)               # back to (L,N)

            s["coeffs"] = coeffs.numpy()

        return s

    def __len__(self):
        return len(self.use_index)


def get_solar_dataloader(
    npz_path: str,
    batch_size: int,
    device,
    val_len: float = 0.1,
    test_len: float = 0.2,
    missing_pattern: str = "block",
    is_interpolate: bool = False,
    num_workers: int = 4,
    target_strategy: str = "random",
    eval_length: int = 24,
):
    dataset = Solar_Dataset(
        npz_path=npz_path,
        mode="train",
        val_len=val_len,
        test_len=test_len,
        missing_pattern=missing_pattern,
        is_interpolate=is_interpolate,
        target_strategy=target_strategy,
        eval_length=eval_length,
        
    )
    train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    dataset_valid = Solar_Dataset(
        npz_path=npz_path,
        mode="valid",
        val_len=val_len,
        test_len=test_len,
        missing_pattern=missing_pattern,
        is_interpolate=is_interpolate,
        target_strategy=target_strategy,
        eval_length=eval_length,
    )
    valid_loader = DataLoader(dataset_valid, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    dataset_test = Solar_Dataset(
        npz_path=npz_path,
        mode="test",
        val_len=val_len,
        test_len=test_len,
        missing_pattern=missing_pattern,
        is_interpolate=is_interpolate,
        target_strategy=target_strategy,
        eval_length=eval_length,
    )
    test_loader = DataLoader(dataset_test, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    scaler = torch.from_numpy(dataset.train_std).to(device).float()
    mean_scaler = torch.from_numpy(dataset.train_mean).to(device).float()

    return train_loader, valid_loader, test_loader, scaler, mean_scaler


