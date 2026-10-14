from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import os

def next_power_of_2(n):
    return 1 << (n - 1).bit_length()

file_path = "/Users/taange001/Documents/Coding/FastHydrology.jl/local_experiments/input/Kazmierczak2024/THWAITES2km_m3_HAB_toto.mat"

data = loadmat(file_path)

# extract the field
B = data["B"]

ny, nx = B.shape

Ny = next_power_of_2(ny)
Nx = next_power_of_2(nx)

pad_y = Ny - ny
pad_x = Nx - nx

pad_top = pad_y // 2
pad_bottom = pad_y - pad_top
pad_left = pad_x // 2
pad_right = pad_x - pad_left

# apply padding
B_padded = np.pad(
    B,
    ((pad_top, pad_bottom), (pad_left, pad_right)),
    mode="constant",
    constant_values=0
)

print("Original shape:", B.shape)
print("Padded shape:", B_padded.shape)

# --- save to Downloads as .npy ---
save_path = os.path.expanduser("~/Downloads/THWAITES_B_padded_512.npy")
np.save(save_path, B_padded)

print(f"Saved to: {save_path}")