"""
Sample usage of the functions in the mrc_parser library
"""

import os

import matplotlib.pyplot as plt

from mrc_parser import MRC_Parser, BoxnetPT, BoxnetTF
from mrc_parser.util import mrc_to_numpy

from tqdm import tqdm

# Config
MRC_DIR = "new-data" # Provide a path to a directory containing .mrc files (and potentially other files)
RESULTS_DIR = "sample-results" # Provide a path to a directory where output images should be saved

if not os.path.exists(RESULTS_DIR):
    os.mkdir(RESULTS_DIR)

# Find file paths
mrc_files = [x for x in os.listdir(MRC_DIR) if x[-4:] == ".mrc"]

# Load model (uncomment model of preference)
model = BoxnetPT("boxnet.pt")
# model = BoxnetTF("boxnet.tflite")

# Move model to device (available for torch only)
if type(model) == BoxnetPT:
    device = "mps" # or "cuda" or "cpu", depending on availability
    model = model.to(device) # Optional

# Init parser
parser = MRC_Parser(model)
results = []

print("Processing files...")

for mrc_file in tqdm(mrc_files):
    mrc_path = os.path.join(MRC_DIR, mrc_file)

    # Boxnet in 1 line
    result = parser(mrc_path)

    # Extract image and masks
    img = mrc_to_numpy(mrc_path)
    particle_mask = result[:, :, 1] # Take channel 1 of all pixels
    bad_mask = result[:, :, 2] # Take channel 2 of all pixels

    results.append((mrc_file, img, particle_mask, bad_mask))

# ================================================================================
# Below is all plotting and doesn't show any special uses of the mrc_parser module
# ================================================================================

print("\nPlotting and saving mask images...")

for result in tqdm(results):
    # Unpack result
    mrc_file, img, particle_mask, bad_mask = result

    # Set up 3 side-by-side plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    for ax in axes:
        ax.set_aspect("auto")
        ax.axis("off")
        ax.imshow(img, cmap="gray")

    # Plot 1: Input image
    axes[0].set_title("Input Image")

    # # Plot 2: Input + Ground Truth Mask
    axes[1].imshow(particle_mask, cmap="jet", alpha=0.2, vmin=0, vmax=1)
    axes[1].set_title("Input + Particle Mask")

    # Plot 3: Input + Predicted Mask
    im = axes[2].imshow(bad_mask, cmap="jet", alpha=0.2, vmin=0, vmax=1)
    axes[2].set_title("Input + Bad Object Mask")

    # Add shared colorbar to the right
    cbar = fig.colorbar(im, ax=axes[2], orientation="vertical", fraction=0.03, pad=0.04)
    cbar.set_label("Mask Intensity", labelpad=10)

    plt.savefig(os.path.join(RESULTS_DIR, f"{mrc_file[:-4]}.jpg"))

