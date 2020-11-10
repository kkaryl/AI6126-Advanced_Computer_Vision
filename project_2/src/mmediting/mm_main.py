from mmcv import Config
from mmcv.runner import set_random_seed
import os.path as osp

from mmedit.datasets import build_dataset
from mmedit.models import build_model
from mmedit.apis import train_model
from mmcv.runner import init_dist

import mmcv
import os

from src.mmediting import mmedit

if __name__ == '__main__':
    cfg = Config.fromfile('configs/restorers/srresnet_srgan/msrresnet_x4c64b16_g1_1000k_div2k.py')
    print(f'Config:\n{cfg.pretty_text}')  # Show the config
    exp_name = '001_MSRResNet_x4_f64b16_DIV2K_1000k_B16G1'

    cfg.work_dir = f'../mm_experiments/{exp_name}'
    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpus = 1

    # Build dataset
    datasets = [build_dataset(cfg.data.train)]

    # Build the SRCNN model
    model = build_model(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    # Create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    # Meta information
    meta = dict()
    if cfg.get('exp_name', None) is None:
        cfg['exp_name'] = osp.splitext(osp.basename(cfg.work_dir))[0]
    meta['exp_name'] = cfg.exp_name
    meta['mmedit Version'] = mmedit.__version__
    meta['seed'] = 0

    print(f"num of parameters in model: \t {sum(p.numel() for p in model.parameters())}")
    print(f"num of parameters we can use: \t 1821085")

    # Train the model
    #train_model(model, datasets, cfg, distributed=True, validate=True, meta=meta)
