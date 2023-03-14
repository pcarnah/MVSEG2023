from pathlib import Path

import numpy as np
import torch
from monai.data import Dataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.transforms import (Compose, LoadImaged, Orientationd, ScaleIntensityd,
                              EnsureChannelFirstd, EnsureTyped, Spacingd, Activations, AsDiscrete,
                              KeepLargestConnectedComponent, SaveImage)

path = Path('/input/')

# Load and segment images
images = [str(p.absolute()) for p in path.glob("*.nii*")]
d = [{"image": im} for im in images]

# Define transforms for image and segmentation
xform = Compose([
    LoadImaged('image'),
    EnsureChannelFirstd('image'),
    Spacingd('image', [0.5,0.5,0.5], diagonal=True, mode='bilinear'),
    Orientationd('image', axcodes='RAS'),
    ScaleIntensityd("image"),
    EnsureTyped('image'),
])

post_tform = Compose(
    [Activations(softmax=True),
     AsDiscrete(argmax=True),
     KeepLargestConnectedComponent(applied_labels=1)
     ]
)

ds = Dataset(d, xform)
loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

net = torch.load('model.md', map_location=torch.device('cpu'))

saver = SaveImage(
    output_dir='/output/',
    output_postfix="seg",
    output_ext=".nii.gz",
    output_dtype=np.uint8,
    separate_folder=False
)

net.eval()
with torch.no_grad():
    for batch in loader:
        out = sliding_window_inference(batch['image'], (96, 96, 96), 16, net)
        out = post_tform(decollate_batch(out))
        meta_dict = decollate_batch(batch["image_meta_dict"])
        for o, m in zip(out, meta_dict):
            saver(o, m)