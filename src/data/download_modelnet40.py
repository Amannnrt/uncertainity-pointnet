"""
Downloads the pre-sampled ModelNet40 point cloud dataset (h5 format, 2048 points/object,
already split into train/test) originally released alongside the PointNet++ codebase.
This is the standard dataset used across nearly all PointNet-family papers, so results
are directly comparable to published baselines.

Usage:
    python src/data/download_modelnet40.py
"""

import os
import shutil
import zipfile
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
ZIP_NAME = "modelnet40_ply_hdf5_2048.zip"

# The original Stanford host is frequently down/times out, so try mirrors in order.
URLS = [
    "https://huggingface.co/datasets/Msun/modelnet40/resolve/main/modelnet40_ply_hdf5_2048.zip",
    "https://github.com/ma-xu/pointMLP-pytorch/releases/download/Modenet40_dataset/modelnet40_ply_hdf5_2048.zip",
    "https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip",
]


def download(url: str, dest: str):
    print(f"Trying {url} ...")

    def _progress(count, block_size, total_size):
        pct = min(int(count * block_size * 100 / total_size), 100)
        print(f"\r  {pct}%", end="", flush=True)

    urllib.request.urlretrieve(url, dest, _progress)
    print("\nDownload complete.")


def download_with_fallback(urls, dest):
    last_err = None
    for url in urls:
        try:
            download(url, dest)
            return
        except Exception as e:
            print(f"\n  Failed ({e}), trying next mirror...")
            last_err = e
    raise RuntimeError(
        f"All download mirrors failed. Last error: {last_err}\n"
        f"Try manually downloading one of:\n" + "\n".join(urls) +
        f"\nand placing the zip at: {dest}"
    )


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, ZIP_NAME)
    extract_dir = os.path.join(DATA_DIR, "modelnet40_ply_hdf5_2048")

    if os.path.exists(extract_dir):
        print(f"Dataset already present at {extract_dir}, skipping.")
        return

    if not os.path.exists(zip_path):
        download_with_fallback(URLS, zip_path)
    else:
        print("Zip already downloaded, skipping download.")

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DATA_DIR)

    os.remove(zip_path)
    print(f"Done. Dataset ready at: {extract_dir}")
    print("Each .h5 file contains 'data' (N x 2048 x 3 point clouds) and 'label' (N,).")
    print("train files: train0.h5 ... train4.h5 | test files: test0.h5, test1.h5")


if __name__ == "__main__":
    main()
