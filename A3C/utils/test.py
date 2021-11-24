from setproctitle import setproctitle as ptitle
import torch
from environment import atari_env
from utils import setup_logger
from model import A3Clstm
from player_util import Agent
from torch.autograd import Variable
import time
import logging
from collections import deque
import numpy as np


def test(args, shared_model, env_conf):
    ptitle('Test Agent')
    gpu_id = args.gpu_ids[-1]
    log = {}
    # setup logger
    setup_logger('{}_log'.format(args.env), r'{0}{1}_log'.format(args.log_dir, args.env))
    # save logger to dictionary(var log)
    log['{}_log'.format(args.env)] = logging.getLogger('{}_log'.format(args.env))
    d_args = vars(args) #函數返回對象object的屬性和屬性值的字典對象。
    for k in d_args.keys():
        # 把args寫入log.logger.info
            # 2021-11-01 13:14:35,117 : lr: 0.0001
            # 2021-11-01 13:14:35,118 : gamma: 0.99
            # 2021-11-01 13:14:35,119 : tau: 1.0
            # 2021-11-01 13:14:35,120 : seed: 1
            # 2021-11-01 13:14:35,121 : workers: 32
            # 2021-11-01 13:14:35,123 : num_steps: 20
            # 2021-11-01 13:14:35,124 : max_episode_length: 10000
        log['{}_log'.format(args.env)].info('{0}: {1}'.format(k, d_args[k]))

    # seed: cpu + cuda
    torch.manual_seed(args.seed)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed)
    env = atari_env(args.env, env_conf, args)
    reward_sum = 0
    start_time = time.time()
    num_tests = 0
    reward_total_sum = 0
    player = Agent(None, env, args, None)
    player.gpu_id = gpu_id
    player.model = A3Clstm(player.env.observation_space.shape[0], player.env.action_space)

    player.state = player.env.reset()
    player.eps_len += 2
    player.state = torch.from_numpy(player.state).float()
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            player.model = player.model.cuda()
            player.state = player.state.cuda()
    flag = True
    max_MA100 = 0
    reward100 = deque(maxlen=100)
    while True:
        if flag:
            # synchronize model
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.model.load_state_dict(shared_model.state_dict())
            else:
                player.model.load_state_dict(shared_model.state_dict())
            player.model.eval()
            flag = False

        player.action_test()
        reward_sum += player.reward # player.reward: r_t
        
        # 死一條命
        if player.done and not player.info:
            state = player.env.reset()
            player.eps_len += 2
            player.state = torch.from_numpy(state).float()
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.state = player.state.cuda()

        # 全死 player.info: self.was_real_done
        elif player.info: 
            flag = True
            num_tests += 1
            reward100.append(reward_sum)
            if len(reward100) != 100:
                MA100 = 0
            else:
                MA100 = np.mean(reward100)
            #reward_total_sum += reward_sum
            #reward_mean = reward_total_sum / num_tests
            log['{}_log'.format(args.env)].info(
                "Time {0}, episode reward {1}, episode length {2}, MA100 {3:.2f}".
                format(
                    time.strftime("%Hh %Mm %Ss",
                                  time.gmtime(time.time() - start_time)),
                    reward_sum, player.eps_len, MA100))
            # 某次episode MA100大於之前max_MA100就儲存模型
            if args.save_max and MA100 >= max_MA100:
                max_MA100 = MA100
                if gpu_id >= 0:
                    with torch.cuda.device(gpu_id):
                        state_to_save = player.model.state_dict()
                        torch.save(state_to_save, '{0}{1}.dat'.format(
                            args.save_model_dir, args.env))
                else:
                    state_to_save = player.model.state_dict()
                    torch.save(state_to_save, '{0}{1}.dat'.format(
                        args.save_model_dir, args.env))

            reward_sum = 0
            player.eps_len = 0
            state = player.env.reset()
            player.eps_len += 2
            time.sleep(10)
            player.state = torch.from_numpy(state).float()
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.state = player.state.cuda()