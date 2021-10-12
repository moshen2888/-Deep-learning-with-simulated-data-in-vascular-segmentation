# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import os
import sys
import tempfile
from glob import glob

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader
from monai.data.image_reader import ITKReader, NibabelReader
from monai import config
from monai.data import ImageDataset, decollate_batch
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
from monai.transforms import Activations, AddChannel, AsDiscrete, Compose, SaveImage, ScaleIntensity, EnsureType


def main(predpath):
    config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)



    images = sorted(glob(os.path.join(predpath, "*.nii.gz")))
    # define transforms for image and segmentation
    imtrans = Compose([ScaleIntensity(), AddChannel(), EnsureType()])
    pred_ds = ImageDataset(images, transform=imtrans, image_only=False, reader=NibabelReader())
    # sliding window inference for one image at every iteration
    pred_loader = DataLoader(pred_ds, batch_size=1, num_workers=1, pin_memory=torch.cuda.is_available())
    post_trans = Compose([EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold_values=True)])
    saver = SaveImage(output_dir="./pred_output", output_ext=".nii.gz", output_postfix="seg")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(
        dimensions=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    model.load_state_dict(torch.load("UnetAdamDice7747.pth"))
    model.eval()
    with torch.no_grad():
        for pred_data in pred_loader:
            pred_images = pred_data[0].to(device)
            # define sliding window size and batch size for windows inference
            roi_size = (128, 128, 128)
            sw_batch_size = 4
            pred_outputs = sliding_window_inference(pred_images, roi_size, sw_batch_size, model)
            pred_outputs = [post_trans(i) for i in decollate_batch(pred_outputs)]
            meta_data = decollate_batch(pred_data[1])
            for pred_output, data in zip(pred_outputs, meta_data):
                saver(pred_output, data)

if __name__ == "__main__":
    predpath = "C:\\Users\\Administrator\\PycharmProjects\\BrainVesselSegmentation\\evalraw"
    main(predpath)