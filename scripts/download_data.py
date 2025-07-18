import os
import requests
import gzip
import shutil
from pathlib import Path

# wrap data downloading process with function
# compatible with goolgle colab 

def download_amazon_data(project_root):
    """
    Downloads and extracts the Amazon co-purchase dataset.

    Args:
        project_root (str): Root directory where the 'data' folder will be created
    """
    project_root = Path(project_root)
    data_dir = project_root / "data"
    # create 'data' folder if not already existed
    os.makedirs(data_dir, exist_ok=True)

    # url link for zipped and unzipped target files
    url = "https://snap.stanford.edu/data/bigdata/amazon/amazon-meta.txt.gz"
    gz_filename = data_dir / "amazon-meta.txt.gz"
    txt_filename = data_dir / "amazon-meta.txt"

    # if un-zipped data file already existed
    # skip all following steps
    if txt_filename.exists():
        print(f"'{txt_filename}' already exists. Skipping download and extraction.")
        return

    # if zipped data file exists, unzip it and write to txt file
    elif gz_filename.exists():
        print(f"'{gz_filename}' found. Extracting...")
        with gzip.open(gz_filename, 'rb') as f_in, open(txt_filename, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        print("Extraction complete.")
        return

    # if neither zipped nor unzipped data file exist
    # start the download process
    print(f"Downloading from {url}...")
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(gz_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("Download complete.")
    except Exception as e:
        print(f"Download failed: {e}")
        return

    # unzip after download
    print("Extracting...")
    try:
        with gzip.open(gz_filename, 'rb') as f_in, open(txt_filename, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        print("Extraction complete.")
    except Exception as e:
        print(f"Extraction failed: {e}")


# # if work will local files
# # 脚本所在的目录 ~/project_root/scripts
# script_dir = Path(__file__).resolve().parent
# # 项目根目录 ~/project_root
# project_root = script_dir.parent

# # 创建 /data 文件夹（如果不存在）
# data_dir = project_root / "data"
# os.makedirs(data_dir, exist_ok=True)

# # 下载和解压的目标文件名
# url = "https://snap.stanford.edu/data/bigdata/amazon/amazon-meta.txt.gz"
# gz_filename = os.path.join(data_dir, "amazon-meta.txt.gz")
# txt_filename = os.path.join(data_dir, "amazon-meta.txt")

# # 如果 txt 已存在就跳过一切
# if os.path.exists(txt_filename):
#     print(f"'{txt_filename}' already exists. Skipping download and extraction.")

# # 如果 gz 存在但 txt 不存在，就只解压
# elif os.path.exists(gz_filename):
#     print(f"'{gz_filename}' found. Extracting...")
#     with gzip.open(gz_filename, 'rb') as f_in, open(txt_filename, 'wb') as f_out:
#         shutil.copyfileobj(f_in, f_out)
#     print("Extraction complete.")

# # 否则开始下载 + 解压
# else:
#     print(f"Downloading from {url}...")
#     with requests.get(url, stream=True) as r:
#         r.raise_for_status()
#         with open(gz_filename, 'wb') as f:
#             for chunk in r.iter_content(chunk_size=8192):
#                 f.write(chunk)
#     print("Download complete.")

#     print("Extracting...")
#     with gzip.open(gz_filename, 'rb') as f_in, open(txt_filename, 'wb') as f_out:
#         shutil.copyfileobj(f_in, f_out)
#     print("Extraction complete.")