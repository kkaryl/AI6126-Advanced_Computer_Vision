from basicsr import train
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--opt', default='options/train/SRResNet_SRGAN/train_MSRResNet_x4.yml')
    parser.add_argument('--opt', default='options/train/CARN/train_CARN_x4.yml')
    # parser.add_argument('--opt', default='options/train/EDSR/train_EDSR_Mx4.yml')
    # parser.add_argument('--opt', default='options/train/SRResNet_SRGAN/train_MSRResNet_x4.yml')
    parser.add_argument('--opt', default='options/train/EDSR/train_EDSR_Mx4 - 2.yml')
    parser.add_argument('--launcher', default='none')
    parser.add_argument('--local_rank', type=int, default=0)
    return parser



if __name__ == '__main__':
    parser = parse_args()
    args = parser.parse_args()
    opt = train.parse_options(args=args)
    train.main(opt)

    # Transforms basicsr/data/transforms.py

    # # Number of workers of reading data for each GPU
    # num_worker_per_gpu: 6
    # # Total training batch size
    # batch_size_per_gpu: 16
