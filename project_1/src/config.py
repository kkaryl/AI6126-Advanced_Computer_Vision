from os.path import join

"""CONSTANTS"""
DATASET_DIR: str = '../data/celeba/'
IMG_DIR: str = join(DATASET_DIR, 'img_align_celeba')
PARTITION_FILE: str = join(DATASET_DIR, 'list_eval_partition.txt')
ATTRIBUTE_FILE: str = join(DATASET_DIR, 'list_attr_celeba.txt')
TRAIN_ATTRIBUTE_LIST: str = join(DATASET_DIR, 'train_attr_list.txt')
VAL_ATTRIBUTE_LIST: str = join(DATASET_DIR, 'val_attr_list.txt')
TEST_ATTRIBUTE_LIST: str = join(DATASET_DIR, 'test_attr_list.txt')
CHECKPOINT_DIR: str = 'checkpoints'
BACKUP_DIR: str = 'backups'

"""HYPER PARAMETERS"""
# Miscs
manual_seed = 42 #1903
evaluate = False
# world_size = 1 #number of distributed processes?
# dist_url = 'tcp://224.66.41.62:23456' #url used to set up distributed training
# dist_backend = 'gloo'
gpu_id = '0'
disable_tqdm = True
auto_hibernate = True

# optimization
train_batch = 256 #256
dl_workers = 8
test_batch = 128 #128
epochs = 50 #60
# start_epoch = 0
lr = 0.1
lr_decay = 'step' #step, cos, linear, linear2exp, schedule
step = 30 # interval for learning rate decay in step mode
schedule = [30, 35, 40, 45, 50, 55, 56, 57, 58, 59, 60] # decrease learning rate at these epochs [150, 225]
turning_point = 100 # epoch number from linear to exponential decay mode
gamma = 0.1 #LR is multiplied by gamma on schedule 0.1
momentum = 0.9
weight_decay = 1e-4
criterion = 'CE'
optimizer = 'Adam'

# Checkpoints and loggers
ckp_resume = '' #path to latest checkpoint (default: none) #join(CHECKPOINT_DIR, 'checkpoint.pth.tar')
ckp_logger_fname = join(CHECKPOINT_DIR, 'log.txt')
checkpoint_fname = join(CHECKPOINT_DIR, 'checkpoint.pth.tar')
bestmodel_fname = join(CHECKPOINT_DIR, 'model_best.pth.tar')
tensorboard_dir = 'runs'
train_plotfig = join(CHECKPOINT_DIR, 'logs.eps')

# Architecture
arch = 'FaceAttrResNet' # #model architecture FaceAttrResNet FaceAttrMobileNetV2 FaceAttrResNeXt
pt_layers = 18 # 34, 50 18
cardinality = 32 #ResNeXt model cardinality (group)
base_width = 4 #ResNeXt model base width (number of channels in each group)
groups = 3 #ShuffleNet model groups
