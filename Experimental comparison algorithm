
#!/usr/bin/python
# -*- coding:utf-8 -*-
#在训练过程采用bagging的主要思想，随机采样
##use the train_advanced.py to test (mapping-based DTL [MK-MMDand adversarial-based DTL[DANN,CDNN])
import argparse
import os
from datetime import datetime
from utils.logger import setlogger
import logging
from utils.train_utils_combines import train_utils

import torch
import numpy as np
import random
import warnings
from torch import nn
print(torch.__version__)
warnings.filterwarnings('ignore')

#集成学习的算法
from torchensemble.voting import VotingClassifier
from torchensemble.fusion import FusionClassifier
from torchensemble.bagging import BaggingClassifier
from torchensemble.gradient_boosting import GradientBoostingClassifier
from torchensemble.snapshot_ensemble import SnapshotEnsembleClassifier
from torchensemble.soft_gradient_boosting import SoftGradientBoostingClassifier
args = None

def display_records(records, logger):
    msg = (
        "{:<28} | Testing Acc: {:.2f} % | Training Time: {:.2f} s |"
        " Evaluating Time: {:.2f} s"
    )

    print("\n")
    for method, training_time, evaluating_time, acc in records:
        logger.info(msg.format(method, acc, training_time, evaluating_time))

def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    # model and data parameters
    # ROBOT ./datasets/Robot_Attitude  # Printer_3D_v2  Dataset_3D
    parser.add_argument('--data_name', type=str, default='ESD', help='the name of the data') #ROBOT
    parser.add_argument('--data_dir', type=str, default='D:/czb/UDTL-master/hvcm',help='the directory of the data')  #./datasets/Robot_Attitude
    parser.add_argument('--checkpoint_dir', type=str,default=r'./23/79/bagging',help='the directory to save the model')
    ##for changing
    parser.add_argument('--model_name', type=str, choices=['cap_multiscale','cap_robot_2400','cnn_features','cnn_robot'],
                        default='cnn_features', help='the name of the model')  #for printer#  cap_multiscale  cnn_multiscale  cap_only_multiscale    #for robot# cap_robot_2400  cnn_robot
    parser.add_argument('--transfer_task', type=list, default=[[1], [2]],help='transfer learning tasks')   # ：args.transfer_task对多关节数据任务的选择
    parser.add_argument('--add_info', type=str, default='__', help=' 为文件名添加额外信息')
    parser.add_argument('--same_condition', type=bool, default=True, help='whether transfering in the same fault condition') ##(same_codition=True)，R75下根据condition_task选择不同的故障程度类型


    parser.add_argument('--bottleneck', type=bool, default=True, help='whether using the bottleneck layer')  ##
    parser.add_argument('--bottleneck_num', type=int, default=256, help='whether using the bottleneck layer')## for bottleneck_layer
    parser.add_argument('--train', type=bool, default=True, help='whether training ')  ##
    parser.add_argument('--last_batch', type=bool, default=True, help='whether using the last batch')
    ###距离
    parser.add_argument('--distance_metric', type=bool, default=True, help='whether use distance metric')
    parser.add_argument('--distance_loss', type=str, choices=['MK-MMD'], default='MK-MMD', help='which distance loss you use')
    parser.add_argument('--trade_off_distance', type=str, default='Step', help='')#'Cons'
    parser.add_argument('--lam_distance', type=float, default=1, help='this is used for Cons')
    ###对抗
    parser.add_argument('--domain_adversarial', type=bool, default=False, help='whether use domain_adversarial')
    parser.add_argument('--adversarial_loss', type=str, choices=['DA', 'CDA'], default='CDA',help='which adversarial loss you use')
    parser.add_argument('--hidden_size', type=int, default=1024, help='在域判别器中的fc的节点数')
    parser.add_argument('--trade_off_adversarial', type=str, default='Step', help='')  ##Step
    parser.add_argument('--lam_adversarial', type=float, default=1, help='this is used for Cons')
    # optimization information
    parser.add_argument('--opt', type=str, choices=['sgd', 'adam'], default='adam', help='the optimizer')
    parser.add_argument('--lr', type=float, default=1e-3, help='the initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='the momentum for sgd')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='the weight decay')
    parser.add_argument('--lr_scheduler', type=str, choices=['step', 'exp', 'stepLR', 'fix'], default='step', help='the learning rate schedule')
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate scheduler parameter for step and exp')
    parser.add_argument('--steps', type=str, default='5, 8', help='the learning rate decay for step and stepLR')#[150,250]
    # training parameters

    parser.add_argument('--cuda_device', type=str, default='0', help='assign device')
    parser.add_argument("--pretrained", type=bool, default=False, help='whether to load the pretrained model')
    parser.add_argument('--batch_size', type=int, default=64, help='batchsize of the training process')
    parser.add_argument('--num_workers', type=int, default=0, help='the number of training process')
    parser.add_argument('--random_seed', type=bool, default=False, help=' 是否使用随机种子跑网络')
    parser.add_argument('--num_routing', type=int, default=2, help='num_routing')
    parser.add_argument('--experiment_num', type=int, default=3, help='the nember of the experiment from 1 to 3')

    # save, load and display information
    parser.add_argument('--n_estimators', type=float, default=2, help='n_estimators')
    parser.add_argument('--max_epoch', type=int, default=100, help='max number of epoch')  #
    parser.add_argument('--middle_epoch', type=int, default=50, help='max number of epoch')
    parser.add_argument('--print_step', type=int, default=100, help='the interval of log training information')

    args = parser.parse_args()
    return args
def set_random_seed(seed=0):
    # seed setting
    print("rand_seed:", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



if __name__ == '__main__':
    for ind in range(1):
        records = []
        args = parse_args()

        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device.strip()

        if args.distance_metric == True:
            dir_str = '_' + args.distance_loss + '_' + args.trade_off_distance

        elif args.domain_adversarial == True:
            dir_str = '_' + args.adversarial_loss + '_' + args.trade_off_adversarial

        if args.data_name=='Printer_3D_v2':
            task_name='_task_'+str(args.condition_task[0][0])+'_'+str(args.condition_task[1][0])+'_'
            transfer_task = '_R75'

            if args.same_condition == True:
                sub_dir = args.model_name + args.add_info  + dir_str + transfer_task + task_name + datetime.strftime(datetime.now(), '%m%d-%H%M%S')
                if args.bottleneck == False:
                    sub_dir = args.model_name +  args.add_info  + dir_str + '_Nobottle' + transfer_task + task_name + datetime.strftime(datetime.now(),'%m%d-%H%M%S')

        elif  args.data_name=='ROBOT':
            task_name = '_task_' + str(args.transfer_task[0][0]) + '_' + str(args.transfer_task[1][0]) + '_'
            if args.bottleneck == False:
                sub_dir = args.model_name + args.add_info  +dir_str + '_Nobottle' + task_name + datetime.strftime(datetime.now(),'%m%d-%H%M%S')
            else:
                sub_dir=args.model_name + args.add_info  +dir_str + task_name + datetime.strftime(datetime.now(),'%m%d-%H%M%S')
        elif args.data_name=="ESD":
            task_name = '_task_' + str(args.transfer_task[0][0]) + '_' + str(args.transfer_task[1][0]) + '_'
            if args.bottleneck == False:
                sub_dir = args.model_name + args.add_info  +dir_str + '_Nobottle' + task_name + datetime.strftime(datetime.now(),'%m%d-%H%M%S')
            else:
                sub_dir=args.model_name + args.add_info  +dir_str + task_name + datetime.strftime(datetime.now(),'%m%d-%H%M%S')

        save_dir = os.path.join(args.checkpoint_dir, sub_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # set the logger
        setlogger(os.path.join(save_dir, 'train.log'))
        # 记录随机种子
        if args.random_seed == True:
            seed = ind
            set_random_seed(seed=seed)  # 如果为true，为每次运行设置随机种子
            logging.info("rand_seed_value:{}".format(seed))

        # save the args
        for k, v in args.__dict__.items():
            logging.info("{}: {}".format(k, v))

        trainer = train_utils(args, save_dir)

        a=trainer.setup()

        model=BaggingClassifier( estimator=a.model_all, n_estimators=args.n_estimators, cuda=True)
        model.set_optimizer("Adam", lr=args.lr, weight_decay=args.weight_decay)
        model.fit(source_train=a.dataloaders['source_train'],target_train=a.dataloaders['target_train'],epochs=args.max_epoch,source_val=a.dataloaders['source_val'],
                  save_dir=save_dir,trainer=trainer,target_val=a.dataloaders['target_val'])

        testing_acc = model.evaluate(test_loader=a.dataloaders['target_val'])
        print(testing_acc)
