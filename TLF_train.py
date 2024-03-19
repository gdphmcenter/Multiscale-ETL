#!/usr/bin/python
# -*- coding:utf-8 -*-

import argparse
import os
from datetime import datetime
from utils.logger import setlogger
import logging
from utils.train_utils_combines import train_utils
import torch
import warnings
print(torch.__version__)
warnings.filterwarnings('ignore')

args = None

def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    # model and data parameters
    parser.add_argument('--model_name', type=str, default='CNN', help='the name of the model') ## Serial_LCN Parral_LCNconcate  Parrral_LCNadd   LSTM  Serial_CLN
    parser.add_argument('--data_name', type=str, default='ESD', help='the name of the data')  #  cnn_1d  Capmulticnn_esd  Cap_only_esd  Serial_rcnn  mutilcnn_add
    parser.add_argument('--data_dir', type=str, default='F:/czb的代码/UDTL-master/hvcm',  help='transfer learning tasks')  #./CWRU
    parser.add_argument('--normlizetype', type=str, default='mean-std',help='the directory of the data')
    parser.add_argument('--transfer_task', type=list, default=[[2], [0]], help='nomalization type')

    # training parameters
    parser.add_argument('--cuda_device', type=str, default='0', help='assign device')
    parser.add_argument('--checkpoint_dir', type=str, default='./24/101/JMMD', help='the directory to save the model') #'./23/datafusion/M/dianci'
    parser.add_argument("--pretrained", type=bool, default=False, help='whether to load the pretrained model是否加载预训练模型')
    parser.add_argument('--batch_size', type=int, default=16, help='batchsize of the training process')
    parser.add_argument('--num_workers', type=int, default=0, help='the number of training process')
    #bottleneck
    parser.add_argument('--bottleneck', type=bool, default=True, help='whether using the bottleneck layer')
    parser.add_argument('--bottleneck_num', type=int, default=256, help='whether using the bottleneck layer')
    parser.add_argument('--last_batch', type=bool, default=True, help='whether using the last batch')

    #distance
    parser.add_argument('--distance_metric', type=bool, default=True, help='whether use distance metric') #距离损失
    parser.add_argument('--distance_loss', type=str, choices=['MK-MMD', 'JMMD', 'CORAL'], default='JMMD', help='which distance loss you use')
    parser.add_argument('--trade_off_distance', type=str, default='Step', help='')
    parser.add_argument('--lam_distance', type=float, default=1, help='this is used for Cons')
    #adversasrial
    parser.add_argument('--domain_adversarial', type=bool, default=False, help='whether use domain_adversarial是否使用域对抗性')
    parser.add_argument('--adversarial_loss', type=str, choices=['DA', 'CDA', 'CDA+E'], default='CDA+E', help='which adversarial loss you use')
    parser.add_argument('--hidden_size', type=int, default=1024, help='whether using the last batch')
    parser.add_argument('--trade_off_adversarial', type=str, default='Step', help='')
    parser.add_argument('--lam_adversarial', type=float, default=1, help='this is used for Cons')

    # optimization information
    parser.add_argument('--opt', type=str, choices=['sgd', 'adam'], default='adam', help='the optimizer')
    parser.add_argument('--lr', type=float, default=1e-3, help='the initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='the momentum for sgd')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='the weight decay')
    parser.add_argument('--lr_scheduler', type=str, choices=['step', 'exp', 'stepLR', 'fix'], default='step', help='the learning rate schedule')
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate scheduler parameter for step and exp')
    parser.add_argument('--steps', type=str, default='150, 250', help='the learning rate decay for step and stepLR')

    # save, load and display information
    parser.add_argument('--middle_epoch', type=int, default=10, help='middle number of epoch')
    parser.add_argument('--max_epoch', type=int, default=20, help='max number of epoch')
    parser.add_argument('--print_step', type=int, default=10, help='the interval of log training information日志训练信息的间隔')

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device.strip()
    # Prepare the saving path for the model
    sub_dir = args.model_name + '_' + str(args.transfer_task[0][0])+str(args.transfer_task[1][0]) + '_' + datetime.strftime(datetime.now(), '%m%d-%H%M%S')#model_name:cnn_features_1d
    save_dir = os.path.join(args.checkpoint_dir, sub_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # set the logger设置记录器
    setlogger(os.path.join(save_dir, 'train.log'))

    # save the args
    for k, v in args.__dict__.items():
        logging.info("{}: {}".format(k, v))

    trainer = train_utils(args, save_dir)
    trainer.setup()
    trainer.train()





