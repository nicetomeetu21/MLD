  # -*- coding:utf-8 -*-
import argparse, os
import torch
import torch.nn as nn

from base.initExperiment import init_experiment
from Trainer.SemiSegTrainer2Dv2 import StandardSegTrainer
from utils.log_function import print_network
from utils.util import find_model_by_iter

from networks.basic_unet import UNet

""" set flags / seeds """
torch.backends.cudnn.benchmark = True

class SegNet(nn.Module):
    def __init__(self):
        super(SegNet, self).__init__()
        self.seg_net = UNet(1, 1, first_channels=64)

    def parameters(self, recurse: bool = True):
        return self.seg_net.parameters()

    def forward(self, input):
        output = self.seg_net(input)
        return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--exp_name', type=str, default='exp name')
    parser.add_argument('--result_root', type=str, default='path to save result')
    parser.add_argument('--visible_devices', type=str, default='3')
    parser.add_argument('--train1_OCT_dir', type=str, default='path to OCT image')
    parser.add_argument('--train1_label_dir', type=str, default='path to label')
    parser.add_argument('--train2_OCT_dir', type=str, default='path to OCT image')
    parser.add_argument('--train2_label_dir', type=str, default='path to label')
    parser.add_argument('--test_OCT_dir', type=str, default='path to OCT image')
    parser.add_argument('--test_label_dir', type=str, default='path to label')
    parser.add_argument('--test_region_mask_dir', type=str, default='path to region_mask')
    # training option
    parser.add_argument('--isTrain', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_iters', type=int, default=50000)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--eval_iters', type=int, default=4000)
    parser.add_argument('--lambda_Dice', type=float, default=1)
    parser.add_argument('--lambda_CE', type=float, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    # test option
    parser.add_argument('--model_iter', type=str, default='50000')
    parser.add_argument('--result_dir', type=str, default=None)
    parser.add_argument('--img_save_dir', type=str, default=None)
    opts = parser.parse_args()
    device = init_experiment(opts, parser)
    # exit()
    seg_net = SegNet()
    if opts.isTrain:
        model = StandardSegTrainer(opts, device, seg_net)
        print_network(model, opts)
        model = model.to(device)
        model.model_train()
    else:
        if opts.result_dir is None:
            opts.result_dir = os.path.join(opts.result_root, opts.exp_name)
        if opts.img_save_dir is None:
            opts.img_save_dir = os.path.join(opts.result_dir, 'test_' + opts.model_iter, 'cubes')
        print('img_save_path:', opts.img_save_dir)
        model = StandardSegTrainer(opts, device, seg_net)
        model_save_path = find_model_by_iter(os.path.join(opts.result_dir, 'checkpoints'), opts.model_iter)
        print('model_save_path: ', model_save_path)
        model.seg_net.load_state_dict(torch.load(model_save_path, map_location=device)['seg'])

        model = model.to(device)
        model.model_test(test_loader=model.test_loader, save_img=True, cal_metric=False, display_img=False, with_pbar=True)