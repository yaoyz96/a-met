
import os
import yaml
import time
import argparse
import numpy as np

import torch
import models
import utils.helpers as helpers
import utils.checkpoints as ckp

from datasets.datasets import SimpleDataLoader
from datasets.datasets import MetaDataLoader
from datasets.metadata_loader import MetaDatasetLoader
from datasets.metadata_loader import BatchDatasetLoader


np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


def main(args):
  if args.mode == "test":
    test_config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

    base = os.path.join(test_config["load_dir"], "config.yaml")
    config = yaml.load(open(base, 'r'), Loader=yaml.FullLoader)
    for k, v in test_config.items():
      config[k] = v

  else:
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    config['_pre_model'] = args.pre

  config['_gpu'], config['_seed'] = args.gpu, args.seed

  if len(args.gpu.split(',')) > 1:
    config['_parallel'] = True

  learner = Learner(args.resume, args.mode, args.manner, config)
  learner.run(config)


def init_log(resume, mode, manner, config):
  """
  Used in training stage.
  Function:
      new training:
        1) create a new checkpoint directory,
        2) save the config.yaml to the new directory,
        3) and create tensorboard writer.
      resume:
        1)
  """
  if resume:
    ckp_dir = config["checkpoint_dir"]

  if mode == "test":
    ckp_dir = config["load_dir"]
  else:
    # create a checkpoint dir and set log path
    sv_params = dict(model=config['model'], data=config['train_dataset'], backbone=config['model_args']['backbone'])
    fs_params = {}

    if manner == "episode":
      fs_params = dict(n_way=config['n_way'], n_shot=config['n_shot'], n_query=config['n_query'])
    ckp_dir = config['ckp_dir'] = helpers.set_checkpoint_dir(**sv_params, **fs_params)
    yaml.dump(config, open(os.path.join(config['ckp_dir'], 'config.yaml'), 'w'))

  helpers.set_log_path(ckp_dir)

  if mode != "test":
    writer = helpers.set_tensorboard(ckp_dir)
    return writer


def init_model(device, manner, model_name, **model_args):
  helpers.print_and_log("\033[0;32mINFO\033[0m |==> Creating {} model '{}'".format(manner, model_name))
  model = models.make(model_name, **model_args).to(device)
  helpers.print_and_log('\033[0;32mINFO\033[0m |==> Num params: {}'.format(helpers.compute_n_params(model)))
  return model


def init_classifier(device, ecd_out_dim, cls_name, cls_args):
  cls_args['in_dim'] = ecd_out_dim
  classifier = models.make(cls_name, **cls_args).to(device)
  return classifier 


def model_parallel(model, config):
  if config.get('_parallel'): 
    model = torch.nn.DataParallel(model)
  return model


def init_data(data_name, config, manner, split):
  """
  Set datasets used for training / validation / testing.
  param data_name
  param manner: "whole", "episode"
  param split: "train", "val", "test"
  """
  dataset = config[split + "_dataset"]

  # init learning mode
  if manner == "whole":
    mode = "batch"    # for batch training
  else:
    mode = "fixed"    # for episodic fixed training

  # specifiy meta-dataset
  if data_name == "meta_dataset":
    if split == "train":
      dataset = ['ilsvrc_2012', 'omniglot', 'aircraft', 'cu_birds', 'dtd', 'quickdraw', 'fungi', 'vgg_flower']  # 8 datasets
    elif split == "val":
      dataset = ['ilsvrc_2012', 'omniglot', 'aircraft', 'cu_birds', 'dtd', 'quickdraw', 'fungi', 'vgg_flower', 'mscoco']  # 9 datasets
    elif split == "test":
      dataset = ["ilsvrc_2012", "omniglot", "aircraft", "cu_birds", "dtd", "quickdraw", "fungi", 
                "vgg_flower", "traffic_sign", "mscoco", "mnist", "cifar10", "cifar100"]  # 10+3 datasets
    mode = "flexible"   # random way/shot/query
    
  # imagenet should be flexible
  if data_name == "ilsvrc_2012" and manner == "episode":
    mode = "flexible"

  return dataset, mode


def init_dataloader(data_name, dataset, data_path, batch_size=1, split="train", manner="episode",
                    n_way=None, n_shot=None, n_query=None, n_episode=None,
                    input_size=84,
                    mode="flexible", device=None,
                  ):
  """
  mode: ["fixed", "flexible"]
    - fixed:
    - flexible:
  manner: ["whole", "episode"]
    - whole: whole set training manner.
    - episode: meta-learning manner, i.e. episodic.
  """
  data_aug = True if split == "train" else False

  if manner == 'episode':
    if data_name == "meta_dataset" or data_name == 'ilsvrc_2012':
      data_loader = MetaDatasetLoader(data_name=data_name,
                                      dataset=dataset,
                                      data_path=data_path,
                                      n_way=n_way,
                                      n_shot=n_shot,
                                      n_query=n_query,
                                      n_episode=n_episode,
                                      mode=mode,
                                      split=split,
                                      device=device,
                                    )
    else:
      data_loader = MetaDataLoader(dataset=dataset,
                                    data_path=data_path,
                                    n_way=n_way,
                                    n_shot=n_shot,
                                    n_query=n_query,
                                    n_episode=n_episode,
                                    input_size=input_size,
                                    batch_size=1,
                                    aug=data_aug,
                                    split=split,
                                    device=device,
                                  )
  elif manner == 'whole':
    if data_name == "ilsvrc_2012":
      data_loader = BatchDatasetLoader(dataset=dataset,
                                      data_path=data_path,
                                      batch_size=batch_size,
                                      #n_iterations=925600//batch_size,
                                      n_iterations=10,
                                      device=device,
                                    )
    else:
      data_loader = SimpleDataLoader(dataset=dataset, 
                                      data_path=data_path, 
                                      batch_size=batch_size, 
                                      input_size=input_size, 
                                      aug=data_aug,
                                      split=split,
                                      device=device,
                                    )
  return data_loader


class Learner:
  def __init__(self, resume, mode, manner, config):
    """
    When training set is 'meta-dataset', it means the whole meta-dataset training data is used, 
    and the training / validation / testing set along with the offical setting, please refer the published paper.

    If use meta-dataset (contains 10 datasets), then use MetaDatasetReader for multiple datasets. 
    """
    self.resume = resume
    self.mode = mode
    self.manner = manner

    # ========== Settings ==========
    self.writer = init_log(resume, mode, manner, config)

    # set avaliable gpu and random seed
    helpers.set_gpu(config['_gpu'])
    helpers.set_seed(config['_seed'])

    # ========= Model and Optimizer =========
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # set main device

    cls_param = dict(cls_name=config['model_args']['classifier'], cls_args=config['model_args']['classifier_args'])
    self.model = init_model(device, manner, config['model'], **config['model_args'])
    self.classifier = init_classifier(device, self.model.out_dim, **cls_param)
    if config.get('freeze_bn'):
      helpers.freeze_bn(self.model)
      print('\033[0;32mINFO\033[0m |==> BN layer freezed!')
    
    if self.mode == 'train_test':
      self.test_model = init_model(device, "episode-test", config['model'], **config['model_args'])

    self.model = model_parallel(self.model, config) if self.mode != "test" else self.model
    self.classifier = model_parallel(self.classifier, config) if self.mode != "test" else self.classifier

    # ========== Dataset ==========
    # If is whole set training, the fs_params will be only used in validation stage.
    fs_params = dict(n_way=config['n_way'], n_shot=config['n_shot'], n_query=config['n_query'])

    if mode == 'train' or mode == 'train_test':
      train_dataset, train_mode = init_data(config['train_dataset'], config, manner, split='train')
      val_dataset, val_mode = init_data(config['val_dataset'], config, manner="episode", split='val')

      # n_episode, mode only used for episodic manner
      self.train_loader = init_dataloader(data_name=config["train_dataset"],
                                          dataset=train_dataset, 
                                          data_path=config['train_dataset_path'],
                                          batch_size=config["batch_size"],
                                          n_episode=config['train_n_episode'],
                                          mode=train_mode,
                                          split="train",
                                          manner=manner,
                                          device=device,
                                          **fs_params,
                                        )                                    
      self.val_loader = init_dataloader(data_name=config["val_dataset"],
                                        dataset=val_dataset, 
                                        data_path=config['val_dataset_path'],
                                        batch_size=config["batch_size"],
                                        n_episode=config['val_n_episode'], 
                                        mode=val_mode,
                                        split="val",
                                        manner="episode",
                                        device=device,
                                        **fs_params,
                                      )
    if mode == 'test' or mode == 'train_test':
      test_dataset, test_mode = init_data(config['test_dataset'], config, manner="episode", split='test')
      self.test_loader = init_dataloader(data_name=config["test_dataset"],
                                        dataset=test_dataset, 
                                        data_path=config['test_dataset_path'], 
                                        batch_size=config["batch_size"],
                                        n_episode=config['test_n_episode'], 
                                        mode=test_mode,
                                        split="test",
                                        manner="episode",
                                        device=device,
                                        **fs_params,
                                      )


  def run(self, config):
    """
    Mode:
      train: training and validation in the training stage.
      test: only testing.
      train_test: training and validation in the training stage, testing after the tranining is done.
      (recommend, automatic do the testing stage)
    """
    start_epoch = -1

    if self.mode == 'train' or self.mode == 'train_test':
      self.train_val(start_epoch, config)

    if self.mode == 'train_test':
      self.test(config['ckp_dir'], config)

    if self.mode == 'test':
      self.test(config['load_dir'], config)


  def train_val(self, start_epoch, config):
    # log
    trlog = helpers.create_log(config)

    # load pretrained model
    if config['_pre_model'] is not None:
      pre_ckp = os.path.join(config["load_dir"], config['_pre_model'])
      helpers.print_and_log("\033[0;32mINFO\033[0m |==> Load pretrained model from: {}".format(pre_ckp))
      self.model, self.classifier = helpers.load_model_sep(self.model, self.classifier, pre_ckp)

    # optimizer
    self.optimizer, self.lr_scheduler = helpers.make_optimizer(params=self.model.parameters(), 
                                                              optimizer_name=config['backbone_optimizer'], 
                                                              **config['bk_optimizer_args'],
                                                            )
    self.cls_optimizer, self.cls_lr_sceduler = helpers.make_optimizer(params=self.classifier.parameters(),
                                                                    optimizer_name=config['classifier_optimizer'], 
                                                                    **config['cls_optimizer_args'],
                                                                  )
    top_models = ckp.TopKModel()

    metric = config['model_args']['meta_args']['metric']
    metaloop = MetaLoopGR(self.model, self.classifier, metric)
    
    for epoch in range(start_epoch+1, config['epoch']):
      start = time.time()

      # train
      self.model.train()
      iter_num = config['train_n_episode'] if self.manner == "episode" else None
      
      train_loss, acc_fs, acc_cls = metaloop.train_loop(self.optimizer, self.cls_optimizer, self.train_loader, iter_num)
      self.writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], epoch)
      self.writer.add_scalar('Train/Loss', train_loss, epoch)
      self.writer.add_scalar('Train/Acc_fs', acc_fs, epoch)
      self.writer.add_scalar('Train/Acc_cls', acc_cls, epoch)
      trlog['train_loss'].append(train_loss)
      trlog['train_acc'].append(acc_fs)

      # eval
      self.model.eval()
      loss, acc = [], []
      iter_num = config['val_n_episode']

      val_loss, val_acc, _ = metaloop.test_loop(self.val_loader, iter_num)
      loss.append(val_loss)
      acc.append(val_acc)

      val_loss = np.array(loss).mean()
      val_acc = np.array(acc).mean()
      self.writer.add_scalar('Val/Loss', val_loss, epoch)
      self.writer.add_scalar('Val/Acc', val_acc, epoch)
      trlog['val_loss'].append(val_loss)
      trlog['val_acc'].append(val_acc)

      # checkpoint
      lr_scheduler_state = self.lr_scheduler.state_dict() if self.lr_scheduler else None
      training = {"epoch": epoch,
                  "bk_optimizer": config["backbone_optimizer"],
                  "optimizer_args": config['bk_optimizer_args'],
                  'optmizer_state': self.optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler_state,
                }
      save_obj = {'config': config,
                  'model': config['model'],
                  'model_args': config['model_args'],
                  'model_state': self.model.state_dict(),
                  'training': training,
                }

      # save model
      top_models.add_top_models(epoch, val_acc, save_obj, config['ckp_dir'])
      ckp.sv_checkpoint(epoch, config['epoch'], config['save_freq'], save_obj, config['ckp_dir'])

      if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            self.cls_lr_sceduler.step()

      if val_acc > trlog["max_acc"]:
        trlog["max_acc"] = val_acc
        trlog["max_acc_epoch"] = epoch
      
      log_str = '\033[0;35m[Epoch {}]\033[0m \033[0;36mtrain\033[0m loss/acc {:.2f}|{:.2f}'.format(epoch, train_loss, acc_fs)
      log_str += ', \033[0;36mval\033[0m loss/acc {:.2f}|{:.2f}'.format(val_loss, val_acc)
      log_str += ', \033[0;36mbest\033[0m acc/epoch {:.2f}|{}'.format(trlog['max_acc'], trlog['max_acc_epoch'])
      log_str += ', use {:.2f} minutes'.format((time.time() - start) / 60)
      helpers.print_and_log(log_str)

      torch.save(trlog, os.path.join(config['ckp_dir'], 'trlog'))

    trlog['top_acc'], trlog['top_acc_epoch'] = top_models.save_top_model(config['ckp_dir'])
    torch.save(trlog, os.path.join(config['ckp_dir'], 'trlog'))


  def test(self, checkpoint_dir, config):
    # load checkpoint
    test_list = ["top1_model.tar", "top2_model.tar", "top3_model.tar", "last_model.tar"]
    metric = config['model_args']['meta_args']['metric']

    for tn in test_list:
      ckp_path = os.path.join(checkpoint_dir, tn)
      helpers.print_and_log('\033[0;32mINFO\033[0m |==> Load model from: {}'.format(ckp_path))
      test_model = self.test_model if self.mode == "train_test" else self.model
      model = helpers.load_model(test_model, ckp_path)

      metaloop = MetaLoopGR(model, self.classifier, metric)
      acc, conf = [], []
      iter_num = config['test_n_episode']

      for i in range(config['test_task_num']):
        test_loss, test_acc, test_conf = metaloop.test_loop(self.test_loader, iter_num)
        helpers.print_and_log('[{}] task {:d} acc = {:.2f} += {:.2f} (%)'.format(tn, i+1, test_acc, test_conf))
        acc.append(test_acc)
        conf.append(test_conf)

      avg_acc = np.mean(acc)
      avg_conf = np.mean(conf)
      helpers.print_and_log('[{}] average test acc for {:d} task = {:.2f} +- {:.2f} (%)'
            .format(tn, config['test_task_num'], avg_acc, avg_conf))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # required
    parser.add_argument('--config', help="path of config file.")

    # system settings
    parser.add_argument('--gpu', default='0', help="available gpu id.")
    parser.add_argument('--seed', default=-1, type=int, help="random seed settings.")

    # resume training
    parser.add_argument('--resume', default=False, action='store_true', help="restart from latest checkpoint.")

    # run mode
    parser.add_argument('--mode', choices=['train', 'test', 'train_test'], default='train_test', help="whether to run training only, testing only, or both training and testing.")
    parser.add_argument('--manner', choices=['whole', 'episode'], default='episode', help="the training manner.")
    
    # pretrain model
    parser.add_argument('--pre', default=None, help="which pretrained model will be load.")
    
    args = parser.parse_args()
    main(args)

    