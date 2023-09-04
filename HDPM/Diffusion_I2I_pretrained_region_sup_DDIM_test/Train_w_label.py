
import os
from typing import Dict

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from Diffusion_DDIM2 import GaussianDiffusionSampler
from dataset_oct_bscans_label import AugmentedDataset_test
from Model import cascadeUNet

def eval(modelConfig: Dict):
    # load model and evaluate
    with torch.no_grad():
        device = torch.device(modelConfig["device"])
        model = cascadeUNet(modelConfig).to(device)
        ckpt = torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["test_load_weight"]), map_location=device)
        model.load_state_dict(ckpt)
        print("model load weight done.")
        model.eval()
        sampler = GaussianDiffusionSampler(
            model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"], modelConfig["skip"]).to(device)

        dataset = AugmentedDataset_test(modelConfig["test_label_path"],modelConfig["test_region_path"], ret_info=True,
                                   cube_names=modelConfig["test_cubenames"],skip_stride=1
                                   )
        dataloader = DataLoader(
            dataset, batch_size=modelConfig["batch_size"], shuffle=False, num_workers=4, pin_memory=True)
        pbar = tqdm(total=len(dataloader))
        for i, (label, region, cubename, imgname) in enumerate(dataloader):
            pbar.update(1)

            num = 0
            for i in range(len(imgname)):
                if os.path.exists(os.path.join(
                    modelConfig["test_dir"], modelConfig["test_load_weight"][:-3], cubename[i], imgname[i])):
                    num+=1
            if num == len(imgname): continue

            label = label.to(device)
            region = region.to(device)

            # Sampled from standard normal distribution
            noisyImage = torch.randn(
                size=[label.shape[0], 1, modelConfig["img_size"][0], modelConfig["img_size"][1]],
                device=device)
            sampledImgs = sampler(noisyImage, label, region)
            sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
            print(sampledImgs.shape)
            for i in range(len(imgname)):
                os.makedirs(os.path.join(modelConfig["test_dir"], modelConfig["test_load_weight"][:-3], cubename[i]),
                            exist_ok=True)
                save_image(torch.cat([sampledImgs[i:i + 1, ...],label[i:i + 1, ...],region[i:i + 1, ...]], dim=0), os.path.join(
                    modelConfig["test_dir"], modelConfig["test_load_weight"][:-3], cubename[i], imgname[i]))
            # exit()
