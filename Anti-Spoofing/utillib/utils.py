from sklearn.preprocessing import LabelEncoder
import cv2
import numpy as np
import os
import sys
import zipfile
import pickle
import requests
import re
import html


le = LabelEncoder()

# image_helpers
# -----------------------------------------------------------

def load_images_with_labels(dir, size=64):
    datas = []
    labels = []
    for label in os.listdir(dir):
        path = os.path.join(dir, label)
        if os.path.isdir(path):
            for file in os.listdir(path):
                try:
                    image_path = os.path.join(path, file)
                    image = cv2.imread(image_path)
                    assert image is not None, "%s is not an image" %file
                    image = cv2.resize(image, (size, size))
                    datas.append(image)
                    labels.append(label)
                except AssertionError as e:
                    pass
        else:
            try:
                image = cv2.imread(path)
                assert image is not None
                image = cv2.resize(image, (size, size))
                datas.append(image)
                labels.append(label)
            except AssertionError as e:
                pass
    datas = np.asarray(datas)
    return datas, labels


# URL helpers
# -----------------------------------------------------------

def is_url(obj: str, allow_file_urls: bool = False) -> bool:
    if not "://" in obj:
        return False
    if allow_file_urls and obj.startswith("file:///"):
        return True
    try:
        res = requests.compat.urlparse(obj)
        if not res.scheme or not res.netloc or not "." in res.netloc:
            return False
        res = requests.compat.urlparse(requests.compat.urljoin(obj, "/"))
        if not res.scheme or not res.netloc or not "." in res.netloc:
            return False
    except:
        return False
    return True


def open_url(url: str, cache_dir: str = None, num_attempts: int = 10, verbose: bool = True, return_path: bool = False):
    """Download the given URL and return a binary-mode file object to access the data."""
    assert is_url(url, allow_file_urls=True)
    assert num_attempts >= 1

    # Handle file URLs.
    if url.startswith('file:///'):
        return open(url[len('file:///'):], "rb")

    # Download.
    url_name = None
    url_data = None
    with requests.Session() as session:
        if verbose:
            print("Downloading %s ..." % url, end="", flush=True)
        for attempts_left in reversed(range(num_attempts)):
            try:
                with session.get(url) as res:
                    res.raise_for_status()
                    if len(res.content) == 0:
                        raise IOError("No data received")

                    if len(res.content) < 8192:
                        content_str = res.content.decode("utf-8")
                        if "download_warning" in res.headers.get("Set-Cookie", ""):
                            links = [html.unescape(link) for link in content_str.split('"') if "export=download" in link]
                            if len(links) == 1:
                                url = requests.compat.urljoin(url, links[0])
                                raise IOError("Google Drive virus checker nag")
                        if "Google Drive - Quota exceeded" in content_str:
                            raise IOError("Google Drive download quota exceeded -- please try again later")

                    match = re.search(r'filename="([^"]*)"', res.headers.get("Content-Disposition", ""))
                    url_name = match[1] if match else url
                    url_data = res.content
                    if verbose:
                        print(" done")
                    break
            except:
                if not attempts_left:
                    if verbose:
                        print(" failed")
                    raise
                if verbose:
                    print(".", end="", flush=True)


def unzip_from_url(data_dir, dataset_url):
    zip_path = open_url(dataset_url, cache_dir='.stylegan2-cache', return_path=True)
    with zipfile.ZipFile(zip_path, 'r') as f:
        f.extractall(data_dir)


# get class by string
# -----------------------------------------------------------
def get_atr_by_name(name: str):
    return getattr()