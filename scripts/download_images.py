#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PIL import Image
from multiprocessing import Pool
from tqdm import tqdm
from urllib3 import PoolManager
from urllib3.util import Retry
import io
import json
import os
import sys
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def download_image(info):
    fname, url = info
    if not os.path.exists(fname):
        http = PoolManager(retries=Retry(connect=3, read=2, redirect=3))
        response = http.request('GET', url)
        image = Image.open(io.BytesIO(response.data))
        image_rgb = image.convert('RGB')
        image_rgb.save(fname, format='jpeg', quality=90)


def parse_dataset(dataset, outdir):
    infos = []
    with open(dataset, 'r') as f:
        data = json.load(f)
        for image in data['images']:
            url = image['url']
            fname = os.path.join(outdir, '{}.jpg'.format(image["imageId"]))
            infos.append((fname, url))
    return infos


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Not enough arguments; exiting!')
        sys.exit(1)

    dataset, outdir = sys.argv[1:]
    if not os.path.exists(outdir):
        print('Making directory...')
        os.makedirs(outdir)

    infos = parse_dataset(dataset, outdir)
    pool = Pool(processes=8)
    with tqdm(total=len(infos)) as progress:
        for _ in pool.imap_unordered(download_image, infos):
            progress.update(1)

    sys.exit(0)
