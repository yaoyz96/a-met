import os
import sys
import cv2
import shutil
import random
import numpy as np
import scipy.stats

import torch
import torch.nn as nn

from PIL import Image
from torchvision.transforms import transforms
from datetime import datetime

from tensorboardX import SummaryWriter
from collections import OrderedDict


##########################
####     Log file     ####
##########################

_log_path = None

def set_log_path(path):
  global _log_path
  _log_path = path


def print_and_log(obj, filename='log.txt'):
  print(obj)
  obj = _del_str(obj)
  if _log_path is not None:
    with open(os.path.join(_log_path, filename), 'a') as f:
      print(obj, file=f)


def _del_str(string):
  del_list = ["\033[0;32m", "\033[0;35m", "\033[0;36m", "\033[0m"]
  for dn in del_list:
    string.replace(dn, '', 3)
  return string


def create_log(config):
    trlog = {}
    trlog['args'] = config
    trlog['train_loss'] = []
    trlog['train_acc'] = []
    trlog['val_loss'] = []
    trlog['val_acc'] = []

    trlog['top_acc'] = []
    trlog['top_acc_epoch'] = []
    trlog['max_acc'] = 0.0
    trlog['max_acc_epoch'] = 0

    return trlog


def read_log(path, stage):
    trlog = {}
    log = torch.load(os.path.join(path, 'trlog'))
    trlog['args'] = log['args']
    trlog['train_loss'] = log['train_loss']
    trlog['train_acc'] = log['train_acc']
    trlog['top_acc'] = log['top_acc']
    trlog['top_acc_epoch'] = log['top_acc_epoch']
    trlog['max_acc'] = log['max_acc']
    trlog['max_acc_epoch'] = log['max_acc_epoch']
    if stage == 'pretrain':
        trlog['val_1shot_loss'] = log['val_1shot_loss']
        trlog['val_5shot_loss'] = log['val_5shot_loss']
        trlog['val_1shot_acc'] = log['val_1shot_acc']
        trlog['val_5shot_acc'] = log['val_5shot_acc']
    elif stage == 'metatrain':
        trlog['val_loss'] = log['val_loss']
        trlog['val_acc'] = log['val_acc']
        trlog['test_loss'] = log['test_loss']
        trlog['test_acc'] = log['test_acc']
    return trlog


###############################
####     Save and Load     ####
###############################

# create save path
def set_checkpoint_dir(model, data, backbone, **fs_params):
  checkpoint_dir = './checkpoints/%s/%s_%s' % (model, data, backbone)
  if fs_params:
    checkpoint_dir += '_%dway_%dshot' % (fs_params['n_way'], fs_params['n_shot'])
  time_logo = datetime.now().strftime("%Y%m%d%H%M")

  checkpoint_dir += '/%s' % (time_logo)
  verify_checkpoint_dir(checkpoint_dir)

  os.makedirs(checkpoint_dir)
  print_and_log('\033[0;32mINFO\033[0m |==> Save model to: %s' % checkpoint_dir)

  return checkpoint_dir


# verify that the checkpoint directory and file exists
def verify_checkpoint_dir(checkpoint_dir, mode="train", resume=False, remove=True):
    # resume training process from lastest checkpoint
    if resume:  
        if not os.path.exists(checkpoint_dir):
            print("\033[0;31mERROR\033[0m |==> Can't resume for checkpoint. Checkpoint directory ({}) does not exist.".format(checkpoint_dir), flush=True)
            sys.exit()
        checkpoint_file = os.path.join(checkpoint_dir, 'checkpoint.pt')
        if not os.path.isfile(checkpoint_file):
            print("\033[0;31mERROR\033[0m |==> Can't resume for checkpoint. Checkpoint file ({}) does not exist.".format(checkpoint_file), flush=True)
            sys.exit()
    # test mode
    elif mode == 'test':
        if checkpoint_dir is None:
            print("\033[0;31mERROR\033[0m |==> Checkpoint dir is empty!", flush=True)
            sys.exit()
        if not os.path.exists(checkpoint_dir):
            print("\033[0;31mERROR\033[0m |==> Checkpoint directory ({}) does not exist.".format(checkpoint_dir), flush=True)
            sys.exit()
    else:
        if os.path.exists(checkpoint_dir):
            if remove and input('\033[0;32mINFO\033[0m |==> {} exists, remove? ([y]/n): '.format(checkpoint_dir)) != 'n':
                shutil.rmtree(checkpoint_dir)
            else:
                print("\033[0;31mERROR\033[0m |==> Checkpoint directory ({}) already exits.".format(checkpoint_dir), flush=True)
                print("\033[0;31mERROR\033[0m |==> If starting a new training run, specify a directory that does not already exist.", flush=True)
                print("\033[0;31mERROR\033[0m |==> If you want to resume a training run, specify the -r option on the command line.", flush=True)
                sys.exit()


def load_model(model, states):
    weights = torch.load(states)['model_state']
    model.load_state_dict(weights, strict=False)
    return model


# for resume training
def load_state(model, optimier, lr_scheduler, states):
    checkpoint = torch.load(states)
    config = checkpoint['config']
    model.load_state_dict(checkpoint['model_state'])
    start_epoch = checkpoint['training']['epoch']
    optimier.load_state_dict(checkpoint['training']['optimizer_state'])
    lr_scheduler.load_state_dict(checkpoint['training']['lr_scheduler'])
    return model, optimier, lr_scheduler, start_epoch, config
    

def load_model_sep(backbone, classifier, states):
    bk_dict = backbone.state_dict()
    cls_dict = classifier.state_dict()

    weights = torch.load(states)['model_state']
    bk_weights = {k: v for k, v in weights.items() if k in bk_dict}
    cls_weights = {k: v for k, v in weights.items() if k in cls_dict}

    bk_dict.update(bk_weights)
    cls_dict.update(cls_weights)
    backbone.load_state_dict(bk_dict)
    classifier.load_state_dict(cls_dict)

    return backbone, classifier


##################################
####     Training helpers     ####
##################################

# set avaliable gpu
def set_gpu(gpu):
    gpu_list = [int(x) for x in gpu.split(',')]
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    if len(gpu_list) != torch.cuda.device_count():
        print("\033[0;31mERROR\033[0m |==> Number of avaiable gpu are not matching to settings!")
        print("\033[0;31mERROR\033[0m |==> GPU setting is {} but got {} gpu available.".format(gpu_list, torch.cuda.device_count()))
        sys.exit()
    print_and_log('\033[0;32mINFO\033[0m |==> Avaliable {} GPU(s) with id(s): {}'.format(torch.cuda.device_count(), gpu_list))


def set_tensorboard(path):
    writer = SummaryWriter(os.path.join(path, 'tensorboard'))
    return writer


# change key name of weights dict
# def rename_state(state):
#     state_dict = OrderedDict()
#     for k, v in state.items():
#         if k[:7] == 'feature':
#             key = 'backbone' + k[7:]   # change prefix 'feature.' to 'backbone.'
#             state_dict[key] = v
#         else:
#             state_dict[k] = v
#     return state_dict


###########################
####     Settings      ####
###########################

# fixed random seed for reproduce the training stage
def set_seed(seed):
    if seed == -1:
        print_and_log("\033[0;32mINFO\033[0m |==> Use Random Seed")
        torch.backends.cudnn.benchmark = True   # optimal
    else:
        print_and_log("\033[0;32mINFO\033[0m |==> Use Fixed Seed: {}".format(seed))
        random.seed(seed)
        np.random.seed(seed)  # numpy
        torch.manual_seed(seed)  # cpu
        torch.cuda.manual_seed(seed)  # gpu
        torch.cuda.manual_seed_all(seed)   # all gpu
        os.environ['PYTHONHASHSEED'] = str(seed)  # fixed hash randomization
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def visualize_dataset(dataset, name, writer, n_samples=16):
    demo = []
    # np.random.choice表示在[0,len(dataset))区间内随机选择n_sample个数值组成一个数组
    for i in np.random.choice(len(dataset), n_samples):
        demo.append(dataset.convert_raw(dataset[i][0])) # 16张标准化后的图片tensor
    # tensorboardX 绘制图片，可用于检查模型的输入，监测feature map的变化，或是观察weight
    writer.add_images('visualize_' + name, torch.stack(demo))
    writer.flush()


def compute_acc(logits, label, reduction='mean'):
    ret = (torch.argmax(logits, dim=1) == label).float()
    if reduction == 'none':
        return ret.detach()
    elif reduction == 'mean':
        return ret.mean().item()


# compute acc for per categroy
def compute_acc_pro(logits, label, category, cat_acc):
    cat_num, num_per_cat = category.shape[0], category.shape[1]
    ret = torch.argmax(logits, dim=1)
    ret, label = ret.view(cat_num, num_per_cat), label.view(cat_num, num_per_cat)
    for i in range(cat_num):
        acc = (ret[i] == label[i]).float().mean()
        cat_acc[category[:,0][i].item()].append(acc.item())
    return cat_acc


def mis_prediction(logits, label, category, cat_mis):
    ret = torch.argmax(logits, dim=1)
    for i in range(len(ret)):
        if ret[i] != label[i]:
            fp = category[ret[i]].item()  # false positive
            tp = category[label[i]].item()  # true positive
            if fp in cat_mis[tp]:
                cat_mis[tp][fp] += 1
            else:
                cat_mis[tp][fp] = 1
    return cat_mis
            

###############################
####     Model helpers     ####
###############################

def mean_confidence_interval(data, confidence=0.95):
    if np.mean(data) > 1:
        data = [x/100 for x in data]
    a = 1.0 * np.array(data)
    n = len(a)
    se = scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return h


def get_mtd_params(method, **kwargs):
    if method == 'baseline':
        mt_params = dict(dropout_rate=kwargs['dropout_rate'])
    elif method == 'deepbdc':
        mt_params = dict(reduce_dim=kwargs['reduce_dim'], t_lr=kwargs['t_lr'])

    return mt_params


def make_optimizer(params, optimizer_name, lr, weight_decay=None, milestones=None, gamma=None):
    if weight_decay is None:
        weight_decay = 0.

    # optimizer
    if optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(params, lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    elif optimizer_name == 'adam':
        optimizer = torch.optim.Adam(params, lr, weight_decay=weight_decay)

    # lr_scheduler
    if milestones:
        lr_params = dict(gamma=gamma, milestones=milestones)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **lr_params)
    else:
        lr_scheduler = None
        
    return optimizer, lr_scheduler


def compute_n_params(model, return_str=True):
    tot = 0
    for p in model.parameters():
        w = 1
        for x in p.shape:
            w *= x
        tot += w
    if return_str:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot


def freeze_bn(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


###############################
####     Visualization     ####
###############################

# draw class activation map
def getCAM(img, fmap, grad):
    '''
    img: image with tensor type [H, W, 3]
    fmap: feature map [feat_dim, H', W']
    grad: gradient [feat_dim, H', W']
    '''
    H, W, _ = img.shape
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    cam = np.zeros(fmap.shape[1:], dtype=np.float32)
    grad = grad.reshape([grad.shape[0], -1])
    weights = np.mean(grad, axis=1)
    for i, w in enumerate(weights):
        cam += w * fmap[i, :, :]
    np.seterr(divide='ignore',invalid='ignore')
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    cam = cv2.resize(cam, (W, H))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    #heatmap[np.where(cam <= 0.3)] = 0
    cam = 0.3 * heatmap + 0.7 * img
    return cam


def transform_convert(img_tensor, transform, resize=False, mode='Tensor'):
    """
    reverse operation: Normalize & ToTensor
    img_tensor: C x H x W
    """
    if 'Normalize' in str(transform):
        normal_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform.transforms))
        mean = torch.tensor(normal_transform[0].mean, dtype=img_tensor.dtype, device=img_tensor.device)
        std = torch.tensor(normal_transform[0].std, dtype=img_tensor.dtype, device=img_tensor.device)
        img_tensor.mul_(std[:,None,None]).add_(mean[:,None,None])

    if resize:
        img_tensor = transforms.Resize(resize=256)(img_tensor)
        
    img_tensor = img_tensor.transpose(0,2).transpose(0,1)  # C x H x W --> H x W x C

    if 'ToTensor' in str(transform) or img_tensor.max() < 1:
        img_tensor = img_tensor.detach()*255
    
    if mode == 'Tensor':
        img = cv2.cvtColor(np.asarray(img_tensor), cv2.COLOR_RGB2BGR)
        return img_tensor

    elif mode == 'PIL':
        if isinstance(img_tensor, torch.Tensor):
            img_tensor = img_tensor.numpy()
        if img_tensor.shape[2] == 3:
            img = Image.fromarray(img_tensor.astype('uint8')).convert('RGB')
        elif img_tensor.shape[2] == 1:
            img = Image.fromarray(img_tensor.astype('uint8')).squeeze()
        else:
            raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(img_tensor.shape[2]))
        return img
    

if __name__ == '__main__':
    log = '/data2/zyy/code/master/checkpoints/baseline-backbone/mini_imagenet_ResNet12-v1_5way_1shot/202206151515/trlog'
    # read log file which is saved by method 'torch.save()'
    log = torch.load(log)
    args = log['args']
    top_acc = log['top_acc']
    top_acc_epoch = log['top_acc_epoch']
    print(f'top_acc: {top_acc}, top_acc_epoch: {top_acc_epoch}')
    