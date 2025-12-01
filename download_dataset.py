import os
import urllib.request
import argparse
from pathlib import Path
import tarfile
import shutil
import json
import sys
import logging

root_logger = logging.getLogger("download_files")
root_logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root_logger.addHandler(handler)

def download_file(url: str, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, open(output_path, "wb") as out_file:
        shutil.copyfileobj(response, out_file)

def extract_archive(archive_path: Path, extract_dir: Path):
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:xz") as tar:
        tar.extractall(path=extract_dir)

        
if __name__ == "__main__":
    with open("object_categories_config.json", "r") as f:
        object_categories_config = json.load(f)

    for key, value in object_categories_config.items():
        if not value["download"]:
            continue

        outpath =  Path("./datasets") / Path("MVTecAD") / Path(f"{key}.tar.xz")
        new_folder = Path("./datasets") / Path("MVTecAD") / Path(f"{key}")

        if not outpath.is_file() and not new_folder.is_dir():
            root_logger.info(msg=f"Downloading file: {key}")
            download_file(value["url"], outpath)
            root_logger.info(msg=f"File {key} downloaded")

        if not new_folder.is_dir():
            root_logger.info(msg=f"Extracting file: {key}")
            extract_archive(outpath, new_folder)
            root_logger.info(msg=f"File: {key} extracted")
        
        if outpath.is_file():
            root_logger.info(msg=f"Cleaning compressed file: {key}")
            os.remove(outpath)
            root_logger.info(msg=f"Compressed File: {key} cleaned")


