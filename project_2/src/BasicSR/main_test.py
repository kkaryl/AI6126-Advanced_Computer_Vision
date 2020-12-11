from basicsr import train, test
import argparse


def parse_args(train=True):
    parser = argparse.ArgumentParser(description="")
    if train:
        # parser.add_argument('--opt', default='options/train/SRResNet_SRGAN/train_MSRResNet_x4.yml')
        parser.add_argument('--opt', default='options/train/CARN/train_CARN_x4.yml')
        # parser.add_argument('--opt', default='options/train/CARN/train_CARNA_x4.yml')
        # parser.add_argument('--opt', default='options/train/EDSR/train_EDSR_Mx4.yml')
        # parser.add_argument('--opt', default='options/train/SRResNet_SRGAN/train_MSRResNetD_x4.yml')
        # parser.add_argument('--opt', default='options/train/RFDN/train_RFDN_x4.yml')
        # parser.add_argument('--opt', default='options/train/IMDN/train_IMDN_x4.yml')
    else:
        # parser.add_argument('--opt', default='options/test/SRResNet_SRGAN/test_MSRResNet_x4.yml')
        parser.add_argument('--opt', default='options/test/SRResNet_SRGAN/test_MSRResNet_x4_woGT.yml')
        # parser.add_argument('--opt', default='options/test/CARN/test_CARN_x4.yml')
        # parser.add_argument('--opt', default='options/test/CARN/test_CARN_x4_woGT.yml')

    parser.add_argument('--launcher', default='none')
    parser.add_argument('--local_rank', type=int, default=0)
    return parser


if __name__ == '__main__':
    is_train = False
    parser = parse_args(train=is_train)
    args = parser.parse_args()
    if is_train:
        opt = train.parse_options(args=args)
        train.main(opt)
    else:
        test_opt = test.parse_options(is_train=is_train, args=args)
        test.main(test_opt)
