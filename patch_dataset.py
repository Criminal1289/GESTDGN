import numpy as np
import torch
from torch.utils.data import Dataset


class PatchDataset(Dataset):
    def __init__(self, patch_files, his_len, pre_len):
        self.samples = []

        for p in patch_files:
            data = np.load(p)  # (T, N, F)
            T, N, F = data.shape

            # 在每一个 patch 内滑窗
            for t in range(T - his_len - pre_len + 1):
                x = data[t: t + his_len]  # (his, N, F)
                y = data[t + his_len: t + his_len + pre_len, :, -1]  # (pre, N)
                self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )
