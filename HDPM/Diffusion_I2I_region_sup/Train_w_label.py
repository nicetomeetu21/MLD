
import os
from typing import Dict

import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from Diffusion import GaussianDiffusionSampler, GaussianDiffusionTrainer
from Model import UNet
from Scheduler import GradualWarmupScheduler
from dataset_for_2d_v2 import OCT2D_multi_Augmented_Dataset
import albumentations as A
def train(modelConfig: Dict):
    device = torch.device(modelConfig["device"])

    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Normalize((0.5,),(0.5,))
    ])
    dataset = OCT2D_multi_Augmented_Dataset(data_roots=['dir to OCT image'
                                                        'dir to region mask'],
                               with_path=False, transform=train_transform)
    dataloader = DataLoader(
        dataset, batch_size=modelConfig["batch_size"], shuffle=True, num_workers=4, drop_last=True, pin_memory=True)

    # model setup
    net_model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
    if modelConfig["training_load_weight"] is not None:
        net_model.load_state_dict(torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["training_load_weight"]), map_location=device))
    optimizer = torch.optim.AdamW(
        net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, multiplier=modelConfig["multiplier"], warm_epoch=modelConfig["epoch"] // 10, after_scheduler=cosineScheduler)
    trainer = GaussianDiffusionTrainer(
        net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)

    label_sample = None
    # start training
    global_iter = 0
    for e in range(modelConfig["epoch"]):
        net_model.train()
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for images, labels in tqdmDataLoader:
                # train
                optimizer.zero_grad()
                x_0 = images.to(device)
                y_0 = labels.to(device)
                if label_sample is None: label_sample = y_0
                loss = trainer(x_0,y_0).sum()/1000
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), modelConfig["grad_clip"])
                optimizer.step()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": loss.item(),
                    "img shape: ": x_0.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
                global_iter += 1
                if global_iter % 2000 == 0: break
        warmUpScheduler.step()
        torch.save(net_model.state_dict(), os.path.join(
            modelConfig["save_weight_dir"], 'ckpt_' + str(e) + "_.pt"))
        net_model.eval()
        with torch.no_grad():
            sampler = GaussianDiffusionSampler(
                net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)
            # Sampled from standard normal distribution
            noisyImage = torch.randn(size=[1, 1, modelConfig["img_size"][0], modelConfig["img_size"][1]], device=device)
            saveNoisy = torch.clamp(noisyImage, 0, 1)
            saveNoisy = torch.cat([saveNoisy, label_sample], dim=0)
            save_image(saveNoisy, os.path.join(
                modelConfig["sampled_dir"], str(e)+modelConfig["sampledNoisyImgName"]), nrow=modelConfig["nrow"])
            sampledImgs = sampler(noisyImage, label_sample)
            sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
            save_image(sampledImgs, os.path.join(
                modelConfig["sampled_dir"], str(e)+modelConfig["sampledImgName"]), nrow=modelConfig["nrow"])
