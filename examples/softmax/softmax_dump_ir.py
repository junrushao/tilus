# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dump per-stage IR for the fused softmax kernel.

Usage:
    python examples/softmax/softmax_dump_ir.py

After running, inspect the IR dumps under .cache-softmax/:
  - Tilus IR passes:  .cache-softmax/programs/<hash>/ir/*.txt
  - Hidet IR passes:  .cache-softmax/programs/<hash>/module/module/ir/*.txt
"""

import os

import tilus
import torch
from tilus import float16, float32, int32
from tilus.utils import cdiv

# ── Configure: cache directory and enable IR dumping ──
tilus.option.cache_dir(os.path.join(os.path.dirname(__file__), ".cache-softmax"))
tilus.option.debug.dump_ir()


# ── Kernel: fused row-wise softmax ──
class FusedSoftmax(tilus.Script):
    def __init__(self):
        super().__init__()
        self.block_m: int = 4
        self.block_n: int = 256
        self.warps: int = 4

    def __call__(
        self,
        m_size: int,
        n_size: int32,
        x_ptr: ~float16,
        y_ptr: ~float16,
    ):
        self.attrs.blocks = (cdiv(m_size, self.block_m),)
        self.attrs.warps = self.warps

        offset_m = self.blockIdx.x * self.block_m
        g_x = self.global_view(x_ptr, dtype=float16, shape=[m_size, n_size])
        g_y = self.global_view(y_ptr, dtype=float16, shape=[m_size, n_size])

        # Pass 1: row-wise max
        r_max = self.register_tensor(dtype=float32, shape=[self.block_m, self.block_n], init=float("-inf"))
        for offset_n in range(0, n_size, self.block_n):
            r_x = self.load_global(g_x, offsets=[offset_m, offset_n], shape=[self.block_m, self.block_n]).to(float32)
            r_max = self.maximum(r_max, r_x)
        r_row_max = self.max(r_max, dim=1, keepdim=True)

        # Pass 2: exp(x - max) and row-wise sum
        r_sum = self.register_tensor(dtype=float32, shape=[self.block_m, self.block_n], init=0.0)
        for offset_n in range(0, n_size, self.block_n):
            r_x = self.load_global(g_x, offsets=[offset_m, offset_n], shape=[self.block_m, self.block_n]).to(float32)
            r_exp = self.exp(r_x - r_row_max)
            r_sum = r_sum + r_exp
        r_row_sum = self.sum(r_sum, dim=1, keepdim=True)

        # Pass 3: normalize and store
        for offset_n in range(0, n_size, self.block_n):
            r_x = self.load_global(g_x, offsets=[offset_m, offset_n], shape=[self.block_m, self.block_n]).to(float32)
            r_exp = self.exp(r_x - r_row_max)
            r_y = r_exp / r_row_sum
            self.store_global(g_y, r_y.to(float16), offsets=[offset_m, offset_n])


def main():
    m_size, n_size = 256, 1024
    x = torch.randn(m_size, n_size, dtype=torch.float16, device="cuda")
    y = torch.empty_like(x)

    softmax = FusedSoftmax()
    softmax(m_size, n_size, x, y)

    y_ref = torch.softmax(x.float(), dim=1).half()
    torch.testing.assert_close(y, y_ref, atol=1e-3, rtol=1e-3)
    print("Softmax result matches reference.")

    # ── Print the dump locations ──
    cache = os.path.join(os.path.dirname(__file__), ".cache-softmax", "programs")
    for prog_hash in os.listdir(cache):
        prog_dir = os.path.join(cache, prog_hash)

        tilus_ir = os.path.join(prog_dir, "ir")
        if os.path.isdir(tilus_ir):
            files = sorted(f for f in os.listdir(tilus_ir) if f.endswith(".txt") and f[0].isdigit())
            print(f"\nTilus IR passes ({len(files)} stages) in {tilus_ir}/")
            for f in files:
                print(f"  {f}")

        hidet_ir = os.path.join(prog_dir, "module", "module", "ir")
        if os.path.isdir(hidet_ir):
            files = sorted(f for f in os.listdir(hidet_ir) if f.endswith(".txt") and f[0].isdigit())
            print(f"\nHidet IR passes ({len(files)} stages) in {hidet_ir}/")
            for f in files:
                print(f"  {f}")


if __name__ == "__main__":
    main()
