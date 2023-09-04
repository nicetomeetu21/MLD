import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from Train_w_label import train, eval


def main(model_config = None):
    modelConfig = {
        "state": "train", # or eval
        "epoch": 100,
        "batch_size": 1,
        "T": 800,
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
        "grad_clip": 1.,
        "device": "cuda:0",
        "pretrained_net1_path": 'path to the pretrainted net',
        "save_weight_dir": "path to save weight",
        "sampled_dir": "path to save sampled_dir",
        'train_cube_names': None,
        "sampledNoisyImgName": "NoisyNoGuidenceImgs.png",
        "sampledImgName": "SampledNoGuidenceImgs.png",
        "nrow": 8
        }
    if model_config is not None:
        modelConfig = model_config
    if modelConfig["state"] == "train":
        os.makedirs(modelConfig["save_weight_dir"], exist_ok=True)
        os.makedirs(modelConfig["sampled_dir"], exist_ok=True)
        train(modelConfig)
    else:
        os.makedirs(os.path.join(modelConfig["test_dir"], modelConfig["test_load_weight"][:-3]), exist_ok=True)
        eval(modelConfig)


if __name__ == '__main__':
    main()
