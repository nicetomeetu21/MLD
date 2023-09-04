
Code for the paper "Model-based Label-to-image Diffusion for Semi-
supervised Choroidal Vessel Segmentation"

## Pseudo label generation

To use the pseudo label generation, first run ` .\pseudo_label_gen\niblack.py`, then run `` .\pseudo_label_gen\postprocess.py`.

The region masks can be automatically segmented by Chen's method as refer to the paper.

The following data parameters are required:

```python
src_dir = r'the dir to OCT cubes'
region_mask_dir = r'the dir to region mask'
result_dir = r'the dir to result cubes'
```



## Hierarchical diffusion probabilistic model

To train the HDPM, first train a sub-network $f_u$ by running `.\HDPM\Diffusion_I2I_region_sup\Main.py`.

The following data parameters are required:

```python
# In Main.py
"save_weight_dir": "path to save weight",
"sampled_dir": "path to save sample",

# In Train_w_label.py
dataset = OCT2D_multi_Augmented_Dataset(data_roots=['dir to OCT image'
                                                    'dir to region mask'],
```



Then train the sub-network $f_s$ with the trained $f_u$ , by running`.\HDPM\Diffusion_I2I_pretrained_region_sup\Main.py`.

The following data parameters are required:
```python
# In Main.py
"save_weight_dir": "path to save weight",
"sampled_dir": "path to save sample",
    
# In Train_w_label.py
    dataset = OCT2D_multi_Augmented_Dataset(data_roots=['dir to OCT image',
                                                        'dir to labels',
                                                        'dir to region mask'],
```



To test using HDPM, run `.\HDPM\Diffusion_I2I_pretrained_region_sup_DDIM_test\Main_test.py`, which uses ddim to speeding up sampling.

The following data parameters are required:
```python
# In Main_test.py
"save_weight_dir": "path to saved weight",
"test_dir": "path to save results",
"test_load_weight": "ckpt_99_.pt",
"test_label_path":"path to testing label",
"test_region_path":'path to testing region mask',
```



## Segmentation network

To train and test the segmentation networks with labeled data and generated data, run `.\train_seg\train_unet2d.py`

The following data parameters are required:

```python
parser.add_argument('--exp_name', type=str, default='exp name')
parser.add_argument('--result_root', type=str, default='path to save result')
parser.add_argument('--train1_OCT_dir', type=str, default='path to OCT image')
parser.add_argument('--train1_label_dir', type=str, default='path to label')
parser.add_argument('--train2_OCT_dir', type=str, default='path to OCT image')
parser.add_argument('--train2_label_dir', type=str, default='path to label')
parser.add_argument('--test_OCT_dir', type=str, default='path to OCT image')
parser.add_argument('--test_label_dir', type=str, default='path to label')
parser.add_argument('--test_region_mask_dir', type=str, default='path to region_mask')
parser.add_argument('--isTrain', type=bool, default=True)
```