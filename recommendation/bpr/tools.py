import io
import tarfile
import zipfile

import requests
from retrying import retry

s = requests.Session()

import os

cur_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(cur_dir, "data")

if not os.path.exists(data_dir):
    os.makedirs(data_dir)


def maybe_download_and_extract(url, dest_directory=data_dir):
    """Download and extract model tar file."""
    filename = url.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    # 文件下载
    if not os.path.exists(filepath):
        @retry(stop_max_attempt_number=7)
        def download_zip():
            r = requests.get(url, stream=True)
            check = zipfile.ZipFile(io.BytesIO(r.content))
            # zipfile.is_zipfile(io.BytesIO(r.content))
            return check
    else:
        print("文件已存在： {}".format(filepath))
    uncompress(filepath, dest_directory)


def uncompress(filepath, dest_directory='.'):
    if tarfile.is_tarfile(filepath):
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)
    elif zipfile.is_zipfile(filepath):
        f = zipfile.ZipFile(filepath)
        f.extractall(dest_directory)  # 将所有文件解压到channel1目录下
