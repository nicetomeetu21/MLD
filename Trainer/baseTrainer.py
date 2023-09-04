import torch
import torch.nn as nn
import os
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import json
class BaseTrainer(nn.Module):
    def __init__(self, opts, device):
        super(BaseTrainer, self).__init__()
        self.opts = opts
        self.device = device

        if opts.isTrain:
            self.global_iter = 0

            self.writer = SummaryWriter(os.path.join(opts.result_dir, 'runs'))
            print("track view:", os.path.join(opts.result_dir, 'runs'))

            self.snapshot_dir = os.path.join(opts.result_dir, 'checkpoints')
            if not os.path.exists(self.snapshot_dir): os.makedirs(self.snapshot_dir)

            self.display_losses = dict()
            self.display_imgs = dict()
            self.test_metric_dict = dict()
            self.best_metric_dict = dict()

    def save(self, snapshot_dir, model_suffix):
        # Save generators, discriminators, and optimizers
        seg_name = os.path.join(snapshot_dir, 'seg_%s.pt' % (model_suffix))
        # opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'seg': self.seg_net.state_dict()}, seg_name)
        # torch.save({'seg': self.seg_opt.state_dict()}, opt_name)

    def check_best(self):
        self.is_better = False
        for name in self.test_metric_dict.keys():
            val = np.mean(self.test_metric_dict[name])
            if name not in self.best_metric_dict.keys() or val > self.best_metric_dict[name]:
                self.best_metric_dict[name] = val
                self.is_better = True


    def best_save(self):
        self.check_best()
        if self.is_better:
            self.save_with_metrics()

    def save_pure(self):
        model_suffix = '%06d' % self.global_iter
        self.save(self.snapshot_dir, model_suffix)
    def save_with_metrics(self):
        model_suffix = '%06d' % self.global_iter
        for name in self.test_metric_dict.keys():
            model_suffix += '_%s_%.3f' % (name, np.mean(self.test_metric_dict[name]))
        self.save(self.snapshot_dir, model_suffix)

    def write_loss_and_img(self,ret_metric=True):
        global_iter = self.global_iter
        self.writer.add_scalars('train losses', self.display_losses, global_iter)

        for k, v in self.display_imgs.items():
            self.writer.add_images(k, v, global_iter)

        self.check_best()
        for k in self.test_metric_dict.keys():
            self.writer.add_scalars(k, {'best':self.best_metric_dict[k],
                                        'curr':np.mean(self.test_metric_dict[k])
                                        }, global_iter)

        if ret_metric:
            ret_str = 'best-'
            for k,v in self.best_metric_dict.items():
                ret_str += k + ':%.3f ' % v
            ret_str += ' curr-'
            for k,v in self.test_metric_dict.items():
                ret_str += k + ':%.3f ' % np.mean(v)
            return ret_str

    def save_metrics_to_json(self):
        if not os.path.exists(self.opts.result_dir): os.mkdir(self.opts.result_dir)
        json_name = 'single_fold_result.json'
        json_path = os.path.join(self.opts.result_dir, json_name)
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                results = json.load(f)
        else:
            results = dict()

        def write_to_result(results_dict, iter, metric_dict):
            if iter not in results_dict.keys(): results_dict[iter] = dict()

            for k in metric_dict.keys():
                if k not in results_dict[iter]: results_dict[iter][k] = []
                results_dict[iter][k] += metric_dict[k]

        write_to_result(results, self.global_iter, self.test_metric_dict)

        with open(json_path, 'w') as f:
            json.dump(results, f, indent=4)