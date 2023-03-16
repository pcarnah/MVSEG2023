#!/usr/bin/env python
import argparse
import json
import tarfile
from pathlib import Path

import torch
import monai
from monai.data import Dataset, DataLoader
from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDistanceMetric
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd, ScaleIntensityd, \
    EnsureTyped, AsDiscreted


def untar(directory, tar_filename):
    """Untar a tar file into a directory

    Args:
        directory: Path to directory to untar files
        tar_filename:  tar file path
    """
    with tarfile.open(tar_filename, "r") as tar_o:
        print('extracting')
        tar_o.extractall(path=directory)
        return tar_o.getnames()

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--submissionfile", required=True, help="Submission File")
parser.add_argument("-r", "--results", required=True, help="Scoring results")
parser.add_argument("-g", "--goldstandard", required=True, help="Goldstandard for scoring")

args = parser.parse_args()
prediction_file_status = "SCORED"

submission_files = untar('./', args.submissionfile)
submission_files = [f for f in submission_files if f.endswith('nii.gz') or f.endswith('nii')]
if not submission_files:
    raise Exception("No submission files found")

print(submission_files)

gt_files = untar('./', args.goldstandard)
gt_files = [f for f in gt_files if f.endswith('nii.gz') or f.endswith('nii')]
if not gt_files:
    raise Exception("No goldstandard files found")

print(gt_files)

submission_files.sort()
gt_files.sort()

# Load and segment images
d = [{"label": label, "pred": pred} for label, pred in zip(gt_files, submission_files)]

keys = ('label', 'pred')

# Define transforms for image and segmentation
xform = Compose([
    LoadImaged(keys),
    EnsureChannelFirstd(keys),
    EnsureTyped(keys),
    Spacingd(keys, [0.25,0.25,0.25], diagonal=True, mode='bilinear'),
    Orientationd(keys, axcodes='RAS'),
    AsDiscreted(keys, to_onehot=2, rounding="torchrounding"),
])

ds = Dataset(d, xform)
loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

mean_dice = DiceMetric(include_background=False)
hd = HausdorffDistanceMetric(percentile=95, include_background=False)
sd = SurfaceDistanceMetric(include_background=False, symmetric=True)

for batch in loader:
    label = batch['label'].as_tensor()
    pred = batch['pred'].as_tensor()

    mean_dice(pred, label)
    hd(pred, label)
    sd(pred, label)


result = {'mean_dice': mean_dice.aggregate().item(),
          '95_hausdorff_distance': hd.aggregate().item() / 4,
          'mean_surface_distance': sd.aggregate().item() / 4,
          'submission_status': prediction_file_status}
with open(args.results, 'w') as o:
  o.write(json.dumps(result))