#!/usr/bin/env python3
"""Utilities for CMD competition data preparation and submission checks."""

import argparse
import pickle
import shutil
import sys
import zipfile
from pathlib import Path

SENSORS = ('ouster', 'hesai', 'robosense')
SPLITS = ('train', 'val', 'test')
CLASSES = {'Car', 'Truck', 'Pedestrian', 'Cyclist'}
REQUIRED_PREDICTION_KEYS = {'name', 'score', 'boxes_lidar', 'pred_labels', 'frame_id'}


def read_split(root, split):
    split_file = root / 'ImageSets' / f'{split}.txt'
    if not split_file.is_file():
        raise FileNotFoundError(f'Missing split file: {split_file}')
    return [line.strip() for line in split_file.read_text().splitlines() if line.strip()]


def check_data(root, sensors, strict):
    root = root.resolve()
    errors = []
    warnings = []
    split_sequences = {split: read_split(root, split) for split in SPLITS}

    for calib_sensor in sensors:
        calib = root / 'calib_to_ego' / f'transformation_matrix_{calib_sensor}_ego.txt'
        if not calib.is_file():
            errors.append(f'missing calibration: {calib.relative_to(root)}')

    split_sensors = {
        'train': [sensor for sensor in sensors if sensor in ('ouster', 'hesai')],
        'val': [sensor for sensor in sensors if sensor in ('ouster', 'hesai')],
        'test': [sensor for sensor in sensors if sensor == 'robosense'],
    }
    for split, sequences in split_sequences.items():
        for seq in sequences:
            seq_dir = root / f'seq{seq}'
            if not seq_dir.is_dir():
                errors.append(f'missing sequence directory: seq{seq}')
                continue
            for sensor in split_sensors[split]:
                sensor_dir = seq_dir / sensor
                count = len(list(sensor_dir.glob('*.npy'))) if sensor_dir.is_dir() else 0
                if count != 200:
                    errors.append(f'seq{seq}/{sensor}: expected 200 npy files, found {count}')

            # Labels are required for the labelled mechanical-source train/val
            # frames, but not for the hidden target-domain test split.
            if split in ('train', 'val'):
                label_dir = (root / 'label_ego' / f'seq{seq}')
                if not label_dir.is_dir():
                    label_dir = root / 'label' / f'seq{seq}'
                count = len(list(label_dir.glob('*.json'))) if label_dir.is_dir() else 0
                if count != 200:
                    errors.append(f'label/seq{seq}: expected 200 json files, found {count}')

    expected_test = len(split_sequences['test']) * 200
    print(f'Dataset root: {root}')
    print('Sequences: ' + ', '.join(f'{key}={len(value)}' for key, value in split_sequences.items()))
    print(f'Expected competition predictions: {expected_test}')
    for warning in warnings:
        print(f'WARNING: {warning}')
    if errors:
        print(f'Found {len(errors)} data issue(s):', file=sys.stderr)
        for error in errors[:100]:
            print(f'  - {error}', file=sys.stderr)
        if len(errors) > 100:
            print(f'  ... and {len(errors) - 100} more', file=sys.stderr)
        return 1 if strict else 0
    print('Data check passed.')
    return 0


def make_infos(root, sensors, workers):
    root = root.resolve()
    if check_data(root, sensors, strict=True):
        raise SystemExit('Dataset is incomplete; resume the download before generating infos.')
    import yaml
    from easydict import EasyDict
    from pcdet.datasets.xmu.xmu_dataset import XMechanismUnmanned
    from pcdet.utils import common_utils

    repo_root = Path(__file__).resolve().parents[1]
    cfg_file = repo_root / 'tools/cfgs/dataset_configs/xmu/xmu_dataset.yaml'
    dataset_cfg = EasyDict(yaml.safe_load(cfg_file.read_text()))
    dataset = XMechanismUnmanned(
        dataset_cfg=dataset_cfg,
        class_names=['Car', 'Truck', 'Pedestrian', 'Cyclist'],
        root_path=root,
        training=True,
        logger=common_utils.create_logger(),
    )

    jobs = []
    for sensor in sensors:
        if sensor in ('ouster', 'hesai'):
            jobs.extend([(sensor, 'train', True), (sensor, 'val', True)])
        elif sensor == 'robosense':
            jobs.append((sensor, 'test', False))

    for sensor, split, has_label in jobs:
        dataset.set_split(split)
        infos = dataset.get_infos(
            dataset.class_names, num_workers=workers, has_lable=has_label,
            sample_seq_list=None, num_features=4, sensor=sensor
        )
        output = root / f'xmu_infos_{split}_{sensor}.pkl'
        with output.open('wb') as handle:
            pickle.dump(infos, handle)
        print(f'Saved {len(infos)} samples to {output}')


def normalize_frame_id(frame_id):
    if isinstance(frame_id, dict):
        return str(frame_id.get('seq', '')).zfill(2), str(frame_id.get('frame', frame_id.get('frame_id', ''))).zfill(4)
    if isinstance(frame_id, (tuple, list)) and len(frame_id) == 2:
        return str(frame_id[0]).zfill(2), str(frame_id[1]).zfill(4)
    raise ValueError(f'Unsupported frame_id: {frame_id!r}')


def validate_predictions(predictions, expected_frames=None):
    import numpy as np

    if not isinstance(predictions, list):
        raise TypeError('prediction.pkl must contain a list of per-frame dictionaries')
    if expected_frames is not None and len(predictions) != expected_frames:
        raise ValueError(f'Expected {expected_frames} frames, found {len(predictions)}')

    seen_frames = set()
    total_boxes = 0
    for index, prediction in enumerate(predictions):
        if not isinstance(prediction, dict):
            raise TypeError(f'Prediction {index} is not a dictionary')
        missing = REQUIRED_PREDICTION_KEYS - prediction.keys()
        if missing:
            raise KeyError(f'Prediction {index} misses keys: {sorted(missing)}')

        frame_key = normalize_frame_id(prediction['frame_id'])
        if frame_key in seen_frames:
            raise ValueError(f'Duplicate frame_id: {frame_key}')
        seen_frames.add(frame_key)

        names = np.asarray(prediction['name'])
        scores = np.asarray(prediction['score'])
        boxes = np.asarray(prediction['boxes_lidar'])
        labels = np.asarray(prediction['pred_labels'])
        count = len(names)
        if scores.shape != (count,) or labels.shape != (count,):
            raise ValueError(f'Prediction {index} has inconsistent vector lengths')
        if boxes.shape != (count, 7):
            raise ValueError(f'Prediction {index} boxes_lidar must have shape (N, 7), got {boxes.shape}')
        if not np.isfinite(scores).all() or not np.isfinite(boxes).all():
            raise ValueError(f'Prediction {index} contains NaN or Inf')
        if ((scores < 0) | (scores > 1)).any():
            raise ValueError(f'Prediction {index} has scores outside [0, 1]')
        unknown_classes = set(names.tolist()) - CLASSES
        if unknown_classes:
            raise ValueError(f'Prediction {index} has unknown classes: {sorted(unknown_classes)}')
        total_boxes += count
    return total_boxes


def package_submission(result, output, data_root=None):
    result = result.resolve()
    output = output.resolve()
    if not result.is_file():
        raise FileNotFoundError(result)
    with result.open('rb') as handle:
        predictions = pickle.load(handle)

    expected_frames = None
    if data_root is not None:
        expected_frames = len(read_split(data_root.resolve(), 'test')) * 200
    total_boxes = validate_predictions(predictions, expected_frames=expected_frames)

    output.parent.mkdir(parents=True, exist_ok=True)
    prediction_file = output.parent / 'prediction.pkl'
    shutil.copy2(result, prediction_file)
    try:
        with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            archive.write(prediction_file, arcname='prediction.pkl')
    finally:
        prediction_file.unlink(missing_ok=True)
    print(f'Validated {len(predictions)} frames and {total_boxes} boxes.')
    print(f'Created submission: {output}')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)

    check = subparsers.add_parser('check-data', help='check whether the dataset download is complete')
    check.add_argument('--data-root', type=Path, default=Path('.'))
    check.add_argument('--sensors', nargs='+', choices=SENSORS, default=list(SENSORS))
    check.add_argument('--strict', action='store_true')

    infos = subparsers.add_parser('make-infos', help='validate data and generate OpenPCDet info files')
    infos.add_argument('--data-root', type=Path, default=Path('.'))
    infos.add_argument('--sensors', nargs='+', choices=SENSORS, default=list(SENSORS))
    infos.add_argument('--workers', type=int, default=4)

    package = subparsers.add_parser('package', help='validate result.pkl and make submission.zip')
    package.add_argument('--result', required=True, type=Path)
    package.add_argument('--output', type=Path, default=Path('submission.zip'))
    package.add_argument('--data-root', type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == 'check-data':
        raise SystemExit(check_data(args.data_root, args.sensors, args.strict))
    if args.command == 'make-infos':
        make_infos(args.data_root, args.sensors, args.workers)
        return
    if args.command == 'package':
        package_submission(args.result, args.output, args.data_root)
        return


if __name__ == '__main__':
    main()
