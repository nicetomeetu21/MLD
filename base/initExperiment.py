import os
import sys
import shutil
import logging
import torch
from utils.log_function import print_options
from natsort import natsorted
def init_experiment(opts, parser):
    """ device configuration """
    if torch.cuda.is_available():
        device = torch.device('cuda:'+str(opts.visible_devices))
    else:
        device = torch.device('cpu')
        logging.warning('cuda is unavailable!')


    if opts.isTrain:
        opts.result_dir = os.path.join(opts.result_root, opts.exp_name)
        if not os.path.exists(opts.result_dir):
            os.makedirs(opts.result_dir)
            code_dir = os.path.abspath(os.path.dirname(os.getcwd()))
            print_options(parser, opts)
            shutil.copytree(code_dir, os.path.join(opts.result_dir, 'code'))
        else:
            sys.exit("result_dir exists: "+opts.result_dir)

    return device