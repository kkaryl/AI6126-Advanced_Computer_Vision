import math
import os
import torchvision.utils

from basicsr.data import create_dataloader, create_dataset
import src.BasicSR.basicsr.data.augments as augments


def main():
    opt = {}
    opt['dist'] = False
    opt['phase'] = 'train'

    opt['name'] = 'DIV2K'
    opt['type'] = 'PairedImageDataset'
    opt['dataroot_gt'] = '../../../data/DIV2K/Train/HR_sub.lmdb'
    opt['dataroot_lq'] = '../../../data/DIV2K/Train/LR_x2_sub.lmdb'
    opt['io_backend'] = dict(type='lmdb')

    opt['gt_size'] = 128
    opt['use_flip'] = True
    opt['use_rot'] = True

    opt['use_shuffle'] = True
    opt['num_worker_per_gpu'] = 1
    opt['batch_size_per_gpu'] = 4
    opt['scale'] = 2

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
        if i > 7:
            break
        print(i)

        lq = data['lq']
        gt = data['gt']
        lq_path = data['lq_path']
        gt_path = data['gt_path']
        print(lq_path, gt_path)

        if opt['use_cutblur']:
            gt_cutblur, lq_cutblur = augments.cutblur(gt, lq, prob=1, alpha=0.7)
            torchvision.utils.save_image(
                lq_cutblur,
                f'tmp/{i:03d}_lq_cutblur.png',
                nrow=nrow,
                padding=padding,
                normalize=False)
            torchvision.utils.save_image(
                gt_cutblur,
                f'tmp/{i:03d}_gt_cutblur.png',
                nrow=nrow,
                padding=padding,
                normalize=False)
        if opt['use_rgb_perm']:
            gt_rgb, lq_rgb = augments.rgb(gt, lq, prob=1)
            torchvision.utils.save_image(
                lq_rgb,
                f'tmp/{i:03d}_lq_rgb.png',
                nrow=nrow,
                padding=padding,
                normalize=False)
            torchvision.utils.save_image(
                gt_rgb,
                f'tmp/{i:03d}_gt_rgb.png',
                nrow=nrow,
                padding=padding,
                normalize=False)

        torchvision.utils.save_image(
            lq,
            f'tmp/{i:03d}_lq.png',
            nrow=nrow,
            padding=padding,
            normalize=False)
        torchvision.utils.save_image(
            gt,
            f'tmp/{i:03d}_gt.png',
            nrow=nrow,
            padding=padding,
            normalize=False)


if __name__ == '__main__':
    main()
