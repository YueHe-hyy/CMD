#!/usr/bin/env python3
"""Download only the CMD files required by competition track 2.

This downloader uses exact resolve URLs instead of the Hugging Face repository
tree API.  It therefore avoids the list_repo_tree pagination that frequently
hits HTTP 429 on large repositories and mirrors.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = 'https://hf-mirror.com/datasets/jinhaodeng/CMD/resolve/main'
FRAMES = tuple(f'{index:04d}' for index in range(200))


def read_split(root, split):
    path = root / 'ImageSets' / f'{split}.txt'
    if not path.is_file():
        raise FileNotFoundError(
            f'Missing {path}. Download ImageSets files first or use --metadata-only.'
        )
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def metadata_paths():
    yield from (f'ImageSets/{split}.txt' for split in ('train', 'val', 'test'))
    yield from (
        f'calib_to_ego/transformation_matrix_{sensor}_ego.txt'
        for sensor in ('ouster', 'hesai', 'robosense')
    )


def required_paths(root, include_labels=True):
    train_val = read_split(root, 'train') + read_split(root, 'val')
    test = read_split(root, 'test')

    if include_labels:
        for seq in train_val:
            for frame in FRAMES:
                yield f'label/seq{seq}/{frame}.json'

    for seq in train_val:
        for sensor in ('ouster', 'hesai'):
            for frame in FRAMES:
                yield f'seq{seq}/{sensor}/{frame}.npy'

    for seq in test:
        for frame in FRAMES:
            yield f'seq{seq}/robosense/{frame}.npy'


def remote_url(base_url, relative_path):
    encoded = '/'.join(quote(part) for part in relative_path.split('/'))
    return f'{base_url.rstrip("/")}/{encoded}?download=true'


def download_one(base_url, root, relative_path, retries, timeout):
    destination = root / relative_path
    if destination.is_file() and destination.stat().st_size > 0:
        return 'skipped'

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + '.part')

    for attempt in range(1, retries + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {'User-Agent': 'cmd-competition-downloader/1.0'}
        if offset:
            headers['Range'] = f'bytes={offset}-'
        request = Request(remote_url(base_url, relative_path), headers=headers)
        print(
            f'GET {relative_path} attempt={attempt}/{retries} resume_bytes={offset}',
            flush=True,
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                status = getattr(response, 'status', response.getcode())
                mode = 'ab' if offset and status == 206 else 'wb'
                with partial.open(mode) as handle:
                    while True:
                        chunk = response.read(8 * 1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
            if partial.stat().st_size == 0:
                raise IOError('downloaded an empty file')
            os.replace(partial, destination)
            return 'downloaded'
        except HTTPError as exc:
            if exc.code == 404:
                raise FileNotFoundError(f'Remote file not found: {relative_path}') from exc
            retry_after = exc.headers.get('Retry-After') if exc.headers else None
            delay = int(retry_after) if retry_after and retry_after.isdigit() else min(300, 5 * attempt)
            print(
                f'HTTP {exc.code} for {relative_path}; retry {attempt}/{retries} in {delay}s',
                file=sys.stderr,
            )
        except (URLError, TimeoutError, OSError) as exc:
            delay = min(300, 5 * attempt)
            print(
                f'{type(exc).__name__} for {relative_path}: {exc}; '
                f'retry {attempt}/{retries} in {delay}s',
                file=sys.stderr,
            )
        if attempt < retries:
            time.sleep(delay)
    raise RuntimeError(f'Failed to download {relative_path} after {retries} attempts')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('.'))
    parser.add_argument('--base-url', default=DEFAULT_BASE_URL)
    parser.add_argument('--retries', type=int, default=50)
    parser.add_argument('--timeout', type=int, default=60)
    parser.add_argument('--request-delay', type=float, default=0.1)
    parser.add_argument('--metadata-only', action='store_true')
    parser.add_argument('--skip-labels', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    root = args.data_root.resolve()

    metadata = list(metadata_paths())
    for index, path in enumerate(metadata, 1):
        state = download_one(args.base_url, root, path, args.retries, args.timeout)
        print(f'[metadata {index}/{len(metadata)}] {state}: {path}', flush=True)

    if args.metadata_only:
        return

    paths = list(required_paths(root, include_labels=not args.skip_labels))
    downloaded = 0
    skipped = 0
    for index, path in enumerate(paths, 1):
        state = download_one(args.base_url, root, path, args.retries, args.timeout)
        downloaded += state == 'downloaded'
        skipped += state == 'skipped'
        print(
            f'[{index}/{len(paths)}] {state}: {path} '
            f'(downloaded={downloaded}, existing={skipped})',
            flush=True,
        )
        if state == 'downloaded' and args.request_delay > 0:
            time.sleep(args.request_delay)


if __name__ == '__main__':
    main()
