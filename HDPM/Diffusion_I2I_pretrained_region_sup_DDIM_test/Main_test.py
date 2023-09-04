import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from Train_w_label import eval
from natsort import natsorted

def main(model_config = None):
    modelConfig = {
        "state": "eval", # or eval
        "epoch": 100,
        "batch_size": 6,
        "T": 800,
        "skip": 5,
        "channel": 64,
        "channel_mult": [1, 2, 4, 8],
        "attn": [2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 1e-4,
        "multiplier": 2.,
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "img_size":(640,400),
        "device": "cuda:0",
        "save_weight_dir": "path to saved weight",
        "test_dir": "path to save results",
        "test_load_weight": "ckpt_99_.pt",
        "test_label_path":"path to testing label",
        "test_region_path":'path to testing region mask',
        "test_for_visual": False,
        "test_for_visual_img_name": "1.png",
        "nrow": 8
        }
    if model_config is not None:
        modelConfig = model_config
    modelConfig['test_cubenames'] = natsorted(os.listdir(modelConfig['test_label_path']))[5:]
    # modelConfig['test_cubenames'] = ['10196']

    os.makedirs(os.path.join(modelConfig["test_dir"], modelConfig["test_load_weight"][:-3]), exist_ok=True)
    eval(modelConfig)


if __name__ == '__main__':
    main()
