import os,time,sys
import torch
from prefetch_generator import BackgroundGenerator
import tqdm
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from datasets.dataset_for_2d_v2 import OCT2D_multi_Augmented_Dataset
from datasets.dataset_for_concat import ConcatDataset
from Trainer.baseTrainer import BaseTrainer
from base.metrics_calculation import cal_metrics, cal_masked_metrics
from networks.mynet_parts.dice_loss import DiceLoss
from networks.mynet_parts.scheduler import get_scheduler
from networks.mynet_parts.init_weights import weights_init_UNIT
import albumentations as A

class StandardSegTrainer(BaseTrainer):
    def __init__(self, opts, device, seg_net):
        super(StandardSegTrainer, self).__init__(opts, device)

        self.seg_net = seg_net

        if opts.isTrain:
            self.seg_opt = torch.optim.Adam(self.seg_net.parameters(), lr=opts.lr, betas=(0.5, 0.999))
            self.seg_scheduler = get_scheduler(self.seg_opt, n_epochs=opts.num_iters, offset=0,
                                               decay_start_epoch=opts.num_iters // 2)

            self.criterion_dice = DiceLoss()
            self.criterion_ce = torch.nn.BCEWithLogitsLoss()

            train_transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomCrop(width=400, height=400),
            ])
            train1_dataset = OCT2D_multi_Augmented_Dataset([opts.train1_OCT_dir, opts.train1_label_dir], transform=train_transform)
            train2_dataset = OCT2D_multi_Augmented_Dataset([opts.train2_OCT_dir, opts.train2_label_dir], transform=train_transform)
            train_dataset = ConcatDataset([train1_dataset, train2_dataset], align=False)
            self.train_loader = DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=1)
            test_dataset = OCT2D_multi_Augmented_Dataset([opts.test_OCT_dir, opts.test_label_dir, opts.test_region_mask_dir],
                                                         with_path=True)
            self.test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
            print('train_dataset len:', len(train_dataset), 'test_dataset len:', len(test_dataset))

            self.apply(weights_init_UNIT('gaussian'))
        else:
            test_dataset = OCT2D_multi_Augmented_Dataset([opts.test_OCT_dir, opts.test_label_dir, opts.test_region_mask_dir],
                                                         with_path=True)
            self.test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

    def model_update(self, img_input, img_label):
        img_out = self.seg_net(img_input)

        loss_Dice = self.criterion_dice(img_out, img_label)
        loss_CE = self.criterion_ce(img_out, img_label)
        loss_S_total = loss_Dice * self.opts.lambda_Dice + loss_CE* self.opts.lambda_CE

        self.seg_opt.zero_grad()
        loss_S_total.backward()
        self.seg_opt.step()
        self.update_learning_rate()

        if self.global_iter % self.opts.eval_iters == 0:
            self.display_losses = {'ce': loss_CE, 'dice': loss_Dice, 'S_total': loss_S_total}
            prb_out = torch.sigmoid(img_out)
            imgs = []
            for j in range(min(3, img_input.shape[0])):
                imgs += [img_input[j:j+1,:, :, :], prb_out[j:j+1,:, :, :], img_label[j:j+1,:, :, :]]
            self.display_imgs['imgs_train'] = torch.cat(imgs, dim=0)

    def update_learning_rate(self):
        if self.seg_scheduler is not None:
            self.seg_scheduler.step()

    def model_train(self):
        pbar = tqdm.tqdm(total=self.opts.num_iters)
        self.train()
        while self.global_iter < self.opts.num_iters:
            for _, (data1, data2) in enumerate(BackgroundGenerator(self.train_loader)):
                img1, label1 = data1
                img2, label2 = data2
                real_A = torch.cat([img1, img2], dim=0)
                real_B = torch.cat([label1, label2], dim=0)
                real_A = real_A.to(self.device)
                real_B = real_B.to(self.device)

                self.model_update(real_A, real_B)

                self.global_iter += 1
                if self.global_iter % self.opts.eval_iters == 0:
                    self.model_test(self.test_loader, cal_metric=True, save_img=False, display_img=True)
                    test_metrics = self.write_loss_and_img(ret_metric=True)
                    pbar.set_postfix_str(s=test_metrics)
                    # saving
                    self.save_pure()
                    self.save_metrics_to_json()

                if self.global_iter == self.opts.num_iters:
                    if self.global_iter % self.opts.eval_iters != 0:
                        self.save_pure()
                        self.save_metrics_to_json()
                    sys.exit('Finish training')

                pbar.update(1)
                start_time = time.time()

    def model_test(self, test_loader, save_img=False, display_img=False, cal_metric=True, with_pbar=False):
        if save_img:
            test_img_save_dir = self.opts.img_save_dir
            if not os.path.exists(test_img_save_dir): os.makedirs(test_img_save_dir)
        if cal_metric:
            self.test_metric_dict = dict()
        if with_pbar:
            pbar = tqdm.tqdm(total=len(test_loader))

        self.eval()
        with torch.no_grad():
            for i, ((real_A, real_B, mask), cubenames, imgnames) in enumerate(BackgroundGenerator(test_loader)):
                real_A = real_A.to(self.device)
                real_B = real_B.to(self.device)
                fake_B = self.seg_net(real_A)
                prb_out = torch.sigmoid(fake_B)

                if cal_metric:
                    prb_out=prb_out.to('cpu')
                    real_B=real_B.to('cpu')
                    mask = mask.to('cpu')
                    cal_metrics(pred_out=prb_out, pred_label=real_B, metric_dict=self.test_metric_dict)
                    cal_masked_metrics(pred_out=prb_out, pred_label=real_B, mask=mask, metric_dict=self.test_metric_dict)

                if i == 0 and display_img:
                    real_A = real_A.to('cpu')
                    real_B = real_B.to('cpu')
                    prb_out = prb_out.to('cpu')
                    imgs_test = []
                    for j in range(min(3, real_A.shape[0])):
                        imgs_test += [real_A[j:j+1,:, :, :], real_B[j:j+1,:, :, :], prb_out[j:j+1,:, :, :]]
                    self.display_imgs['imgs_test'] = torch.cat(imgs_test, dim=0)

                if save_img:
                    bin_out = (prb_out > 0.5).float()

                    cubename = cubenames[0]
                    imgname = imgnames[0]

                    img_dir = os.path.join(test_img_save_dir, cubename)
                    if not os.path.exists(img_dir): os.makedirs(img_dir)
                    img_path = os.path.join(img_dir, imgname)
                    save_image(bin_out, img_path)

                if with_pbar:
                    pbar.update(1)
