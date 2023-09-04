
import os
from typing import Dict

import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from Diffusion import GaussianDiffusionSampler, GaussianDiffusionTrainer
from Model import cascadeUNet
from Scheduler import GradualWarmupScheduler
from dataset_for_2d_v2 import OCT2D_multi_Augmented_Dataset
import albumentations as A
def train(modelConfig: Dict):
    device = torch.device(modelConfig["device"])
    # dataset
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Normalize((0.5,),(0.5,))
    ])
    dataset = OCT2D_multi_Augmented_Dataset(data_roots=['dir to OCT image',
                                                        'dir to labels',
                                                        'dir to region mask'],
                               with_path=False, transform=train_transform)
    dataloader = DataLoader(
        dataset, batch_size=modelConfig["batch_size"], shuffle=True, num_workers=4, drop_last=True, pin_memory=True)

    # model setup
    net_model = cascadeUNet(modelConfig).to(device)
    net_model.net1.load_state_dict(torch.load(modelConfig["pretrained_net1_path"], map_location=device))

    for para in net_model.net1.parameters():
        para.requires_grad = False
    print('load from', modelConfig["pretrained_net1_path"])
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, net_model.parameters())
        , lr=modelConfig["lr"], weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, multiplier=modelConfig["multiplier"], warm_epoch=modelConfig["epoch"] // 10, after_scheduler=cosineScheduler)
    trainer = GaussianDiffusionTrainer(
        net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)

    # start training
    global_iter = 0
    for e in range(modelConfig["epoch"]):
        label_sample = None
        img_sample = None
        net_model.train()
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for images, labels, region_labels in tqdmDataLoader:
                # train
                optimizer.zero_grad()
                x_0 = images.to(device)
                y_0 = labels.to(device)
                y_1 = region_labels.to(device)
                if label_sample is None:
                    label_sample = y_0
                    region_label_sample = y_1
                    img_sample = x_0
                loss = trainer(x_0,y_0, y_1).sum()/1000
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
                if global_iter % 400 == 0: break
        warmUpScheduler.step()
        if e % 5 == 0 or e > 90:
            torch.save(net_model.state_dict(), os.path.join(
                modelConfig["save_weight_dir"], 'ckpt_' + str(e) + "_.pt"))
            net_model.eval()
            with torch.no_grad():
                sampler = GaussianDiffusionSampler(
                    net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)
                # Sampled from standard normal distribution
                noisyImage = torch.randn(size=[1, 1, modelConfig["img_size"][0], modelConfig["img_size"][1]], device=device)
                saveNoisy = torch.clamp(noisyImage, 0, 1)
                saveNoisy = torch.cat([saveNoisy, region_label_sample, label_sample, img_sample], dim=0)
                save_image(saveNoisy, os.path.join(
                    modelConfig["sampled_dir"], str(e)+modelConfig["sampledNoisyImgName"]), nrow=modelConfig["nrow"])
                sampledImgs = sampler(noisyImage, label_sample, region_label_sample)
                sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
                save_image(sampledImgs, os.path.join(
                    modelConfig["sampled_dir"], str(e)+modelConfig["sampledImgName"]), nrow=modelConfig["nrow"])
