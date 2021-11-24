import os
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import torch
import torch.multiprocessing as mp
from utils.environment import atari_env
from utils.utils import read_config
from utils.model import A3Clstm
from utils.train import train
from utils.test import test
from utils.shared_optim import SharedRMSprop, SharedAdam
import time

#undo_logger_setup()
parser = argparse.ArgumentParser(description='A3C')
parser.add_argument(
    '--lr',
    type=float,
    default=0.0001,
    metavar='LR',
    help='learning rate (default: 0.0001)')
parser.add_argument(
    '--gamma',
    type=float,
    default=0.99,
    metavar='G',
    help='discount factor for rewards (default: 0.99)')
parser.add_argument(
    '--tau',
    type=float,
    default=1.00,
    metavar='T',
    help='parameter for GAE (default: 1.00)')
parser.add_argument(
    '--seed',
    type=int,
    default=1,
    metavar='S',
    help='random seed (default: 1)')
parser.add_argument(
    '--workers',
    type=int,
    default=4,
    metavar='W',
    help='how many training processes to use (default: 4)')
parser.add_argument(
    '--num-steps',
    type=int,
    default=20,
    metavar='NS',
    help='number of forward steps in A3C (default: 20)')
parser.add_argument(
    '--max-episode-length',
    type=int,
    default=10000,
    metavar='M',
    help='maximum length of an episode (default: 10000)')
parser.add_argument(
    '--env',
    default='Pong-v0',
    metavar='ENV',
    help='environment to train on (default: Pong-v0)')
parser.add_argument(
    '--env-config',
    default='config.json',
    metavar='EC',
    help='environment to crop and resize info (default: config.json)')
parser.add_argument(
    '--shared-optimizer',
    default=True,
    metavar='SO',
    help='use an optimizer without shared statistics.')
parser.add_argument(
    '--load', default=False, metavar='L', help='load a trained model')
parser.add_argument(
    '--save-max',
    default=True,
    metavar='SM',
    help='Save model on every test run high score matched or bested')
parser.add_argument(
    '--optimizer',
    default='Adam',
    metavar='OPT',
    help='shares optimizer choice of Adam or RMSprop')
parser.add_argument(
    '--load-model-dir',
    default='trained_models/',
    metavar='LMD',
    help='folder to load trained models from')
parser.add_argument(
    '--save-model-dir',
    default='trained_models/',
    metavar='SMD',
    help='folder to save trained models')
parser.add_argument(
    '--log-dir', default='logs/', metavar='LG', help='folder to save logs')
parser.add_argument(
    '--gpu-ids',
    type=int,
    default=-1,
    nargs='+',
    help='GPUs to use [-1 CPU only] (default: -1)')
parser.add_argument(
    '--amsgrad',
    default=True,
    metavar='AM',
    help='Adam optimizer amsgrad parameter')
parser.add_argument(
    '--skip-rate',
    type=int,
    default=4,
    metavar='SR',
    help='frame skip rate (default: 4)')
parser.add_argument(
    '--max-train-episode',
    type=int,
    default=100000,
    metavar='MTE',
    help='n episodes to train model per worker (default: 100000)')

# Based on
# https://github.com/pytorch/examples/tree/master/mnist_hogwild
# Training settings
# Implemented multiprocessing using locks but was not beneficial. Hogwild
# training was far superior

if __name__ == '__main__':
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if args.gpu_ids == -1:
        args.gpu_ids = [-1]
    else:
        #Set the method which should be used to start child processes. 
        #method can be 'fork', 'spawn' or 'forkserver'.

        #Note that this should be called at most once, 
        #and it should be protected inside the if __name__ == '__main__' clause of the main module.
        torch.cuda.manual_seed(args.seed)
        # --multi-process step 1--
        mp.set_start_method('spawn')
    setup_json = read_config(args.env_config)
    # 預設config: {'crop1': 34, 'crop2': 34, 'dimension2': 80}
    # 若有特定遊戲，調整config
    env_conf = setup_json["Default"]
    for i in setup_json.keys():
        if i in args.env:
            env_conf = setup_json[i]
    env = atari_env(args.env, env_conf, args)
    # set up a shared model & share memory
    shared_model = A3Clstm(env.observation_space.shape[0], env.action_space)
    if args.load:
        # load trained model
        saved_state = torch.load(
            '{0}{1}.dat'.format(args.load_model_dir, args.env),
            map_location=lambda storage, loc: storage)
        # use the weights of trained parameters to replace shared model's
        shared_model.load_state_dict(saved_state)
    shared_model.share_memory()
    # set up an optimizer & share memory
    if args.shared_optimizer:
        if args.optimizer == 'RMSprop':
            optimizer = SharedRMSprop(shared_model.parameters(), lr=args.lr)
        if args.optimizer == 'Adam':
            optimizer = SharedAdam(
                shared_model.parameters(), lr=args.lr, amsgrad=args.amsgrad)
        optimizer.share_memory()
    else:
        #如果沒有設定shared_optimizer，在train()函數下會自己建立一個獨立的optimizer去訓練
        optimizer = None
    # --multi-process step 2--
    processes = []
    # --multi-process step 3-- mp.Process(fn, args)
    p = mp.Process(target=test, args=(args, shared_model, env_conf)) #test: 在訓練前先進行一次遊戲測試
    # --multi-process step 4-- start to execute fns
    p.start()
    # --multi-process step 5-- 
    processes.append(p)
    time.sleep(0.1)
    for rank in range(0, args.workers):
        #創建多個process，並分別指定到不同的gpu中。
        #用shared_model與shared_optimizer去訓練
        p = mp.Process(
            target=train, args=(rank, args, shared_model, optimizer, env_conf))
        p.start()
        #一有結果就append進processes
        processes.append(p)
        time.sleep(0.1)
    # wait for all processes finish
    for p in processes:
        time.sleep(0.1)
        p.join()