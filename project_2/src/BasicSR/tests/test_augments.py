import math
import os
import torchvision.utils

from basicsr.data import create_dataloader, create_dataset
import src.BasicSR.basicsr.data.augments as augments

def main():
    opt = {}
    opt['dataroot_gt'] = '../../data/DIV2K/Train/HR_sub.lmdb'
    opt['dataroot_lq'] = '../../data/DIV2K/Train/LR_x4_sub.lmdb'
    opt['io_backend'] = dict(type='imdb')

    opt['gt_size'] = 128
    opt['use_flip'] = True
    opt['use_rot'] = True

    opt['use_shuffle'] = True
    opt['num_worker_per_gpu'] = 2
    opt['batch_size_per_gpu'] = 16
    opt['scale'] = 4

    opt['dataset_enlarge_ratio'] = 1

    opt['use_cutblur'] = True
    opt['use_rgb_perm'] = True

    os.makedirs('tmp', exist_ok=True)

    dataset = create_dataset(opt)
    data_loader = create_dataloader(
        dataset, opt, num_gpu=0, dist=opt['dist'], sampler=None)

    nrow = int(math.sqrt(opt['batch_size_per_gpu']))
    padding = 2 if opt['phase'] == 'train' else 0

    print('start...')
    for i, data in enumerate(data_loader):
        if i > 5:
            break
        print(i)

        lq = data['lq']
        gt = data['gt']
        lq_path = data['lq_path']
        gt_path = data['gt_path']
        print(lq_path, gt_path)

        if opt.get('use_cutblur', False):
            gt_cutblur, lq_cutblur = augments.cutblur(gt, lq, prob = 1, alpha = 0.7)
            torchvision.utils.save_image(
                lq_cutblur,
                f'tmp/lq_cutblur_{i:03d}.png',
                nrow=nrow,
                padding=padding,
                normalize=False)
            torchvision.utils.save_image(
                gt_cutblur,
                f'tmp/gt_cutblur_{i:03d}.png',
                nrow=nrow,
                padding=padding,
                normalize=False)
        if opt.get('use_rgb_perm', False):
            gt_rgb, lq_rgb = augments.rgb(gt, lq, prob=1)
            torchvision.utils.save_image(
                lq_rgb,
                f'tmp/lq_rgb_{i:03d}.png',
                nrow=nrow,
                padding=padding,
                normalize=False)
            torchvision.utils.save_image(
                gt_rgb,
                f'tmp/gt_rgb_{i:03d}.png',
                nrow=nrow,
                padding=padding,
                normalize=False)


        torchvision.utils.save_image(
            lq,
            f'tmp/lq_{i:03d}.png',
            nrow=nrow,
            padding=padding,
            normalize=False)
        torchvision.utils.save_image(
            gt,
            f'tmp/gt_{i:03d}.png',
            nrow=nrow,
            padding=padding,
            normalize=False)

if __name__ == '__main__':
    main()