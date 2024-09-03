# file_handler.py
import os
import shutil
import zipfile
import tarfile
import json
import yaml
import csv
import pandas as pd
from typing import Union, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

class FileHandler:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def read_file(self, file_path: str) -> Union[str, Dict[str, Any], pd.DataFrame]:
        _, ext = os.path.splitext(file_path)
        with open(file_path, 'r') as file:
            if ext == '.json':
                return json.load(file)
            elif ext in ['.yaml', '.yml']:
                return yaml.safe_load(file)
            elif ext == '.csv':
                return pd.read_csv(file)
            else:
                return file.read()

    def write_file(self, file_path: str, data: Union[str, Dict[str, Any], pd.DataFrame]) -> None:
        _, ext = os.path.splitext(file_path)
        with open(file_path, 'w') as file:
            if ext == '.json':
                json.dump(data, file, indent=2)
            elif ext in ['.yaml', '.yml']:
                yaml.dump(data, file)
            elif ext == '.csv':
                if isinstance(data, pd.DataFrame):
                    data.to_csv(file, index=False)
                else:
                    csv.writer(file).writerows(data)
            else:
                file.write(data)

    def compress_files(self, file_paths: List[str], output_path: str, method: str = 'zip') -> None:
        if method == 'zip':
            with zipfile.ZipFile(output_path, 'w') as zipf:
                for file in file_paths:
                    zipf.write(file, os.path.basename(file))
        elif method == 'tar':
            with tarfile.open(output_path, 'w:gz') as tarf:
                for file in file_paths:
                    tarf.add(file, arcname=os.path.basename(file))

    def decompress_file(self, file_path: str, output_dir: str) -> None:
        _, ext = os.path.splitext(file_path)
        if ext == '.zip':
            with zipfile.ZipFile(file_path, 'r') as zipf:
                zipf.extractall(output_dir)
        elif ext in ['.tar', '.gz']:
            with tarfile.open(file_path, 'r:*') as tarf:
                tarf.extractall(output_dir)

    def batch_process_files(self, directory: str, process_func: callable) -> None:
        with ThreadPoolExecutor() as executor:
            files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
            list(tqdm(executor.map(process_func, files), total=len(files)))

    def safe_delete(self, path: str) -> None:
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)

    def create_directory_structure(self, structure: Dict[str, Any], base_path: str = '') -> None:
        for key, value in structure.items():
            path = os.path.join(base_path, key)
            if isinstance(value, dict):
                os.makedirs(path, exist_ok=True)
                self.create_directory_structure(value, path)
            else:
                with open(path, 'w') as f:
                    f.write(value or '')