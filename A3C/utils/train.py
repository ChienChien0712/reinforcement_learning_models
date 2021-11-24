from setproctitle import setproctitle as ptitle #設置進程名稱
import torch
import torch.optim as optim
from environment import atari_env
from utils import ensure_shared_grads, setup_logger
from model import A3Clstm
from player_util import Agent
from torch.autograd import Variable
import time
import logging


# rank: 0, 1, 2, ..., n_workers #default=32
def train(rank, args, shared_model, optimizer, env_conf):
    start_time = time.time()
    # set a name for a process
    ptitle('Training Agent: {}'.format(rank)) 
    # set logger
    log = {}
    setup_logger('{0}_train_{1}_log'.format(args.env, rank), r'{0}{1}_train_{2}_log'.format(args.log_dir, args.env, rank))
    log['{0}_train_{1}_log'.format(args.env, rank)] = logging.getLogger('{0}_train_{1}_log'.format(args.env, rank))

    # sent data to certain GPU
    gpu_id = args.gpu_ids[rank % len(args.gpu_ids)] #32個workers去分配給4個gpu(0,1,2,3,0,1,2,3,0,1,2,3,...)
    # set torch(cpu) & torch(cuda) seeds
    torch.manual_seed(args.seed + rank) # each worker was possessed different seed(cpu)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed + rank) # each worker was possessed different seed(cuda)
    # reconstruct env: "env_conf" represents the file called "config.json"
    env = atari_env(args.env, env_conf, args)
    
    # if args.shared_optimizer=False, use memory-unshared optimizer
    if optimizer is None:
        #如果沒有設定shared_optimizer，就會用獨立的optimizer去訓練shared_model 
        if args.optimizer == 'RMSprop':
            optimizer = optim.RMSprop(shared_model.parameters(), lr=args.lr)
        if args.optimizer == 'Adam':
            optimizer = optim.Adam(shared_model.parameters(), lr=args.lr, amsgrad=args.amsgrad)
            
    #set env seed
    env.seed(args.seed + rank)
    # for certain rank, create an Agent as "player"
    # player have following attributes:
        #self.model
        #self.env = env
        #self.state = state
        #self.hx = None
        #self.cx = None
        #self.eps_len = 0
        #self.args = args
        #self.values = []
        #self.log_probs = []
        #self.rewards = []
        #self.entropies = []
        #self.done = True
        #self.info = None #True: end the game #False: otherwise
        #self.reward = 0
        #self.gpu_id = -1
    # player have following functions:
        #1. action_train()
        #2. action_test()
        #3. clear_actions()
    
    # create a player, aka worker or agent
    player = Agent(None, env, args, None)
    player.gpu_id = gpu_id
    # player.env.observation_space.shape = (1, 80, 80) #gray, height, width
    player.model = A3Clstm(player.env.observation_space.shape[0], player.env.action_space)

    player.state = player.env.reset() #initial state
    player.state = torch.from_numpy(player.state).float()
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            player.state = player.state.cuda() # state: -->cuda
            player.model = player.model.cuda() # model: -->cuda
    player.model.train() #set model into train mode
    player.eps_len += 2
    episode_passed = 0
    while True:
        if player.done:
            log['{0}_train_{1}_log'.format(args.env, rank)].info("worker {0} episode {1}".format(rank, episode_passed))
        # syncronize player.model's weights of parameters by shared_model's
        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                player.model.load_state_dict(shared_model.state_dict())
        else:
            player.model.load_state_dict(shared_model.state_dict())
        # initial cx, hx: both are zero vector
        if player.done:
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.cx = Variable(torch.zeros(1, 512).cuda())
                    player.hx = Variable(torch.zeros(1, 512).cuda())
            else:
                player.cx = Variable(torch.zeros(1, 512))
                player.hx = Variable(torch.zeros(1, 512))
        else:
            player.cx = Variable(player.cx.data)
            player.hx = Variable(player.hx.data)
        
        # player.action_train(): take args.num_steps(defualt=20) steps of actions at most
        # save the results of interactions to following player's attributes:
            #1. player.values
            #2. player.log_probs
            #3. player.rewards
            #4. player.entropies
        for step in range(args.num_steps):
            player.action_train()
            if player.done:
                episode_passed += 1
                break
        
        # if player.done, env will be reset
        if player.done:
            state = player.env.reset()
            player.state = torch.from_numpy(state).float()
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.state = player.state.cuda()
        
        # R: V(s_{last+1})
        # if "done" within args.num_steps, set R as 0.
        # else if "not done", set R as V.
        R = torch.zeros(1, 1)
        if not player.done:
            value, _, _ = player.model((Variable(player.state.unsqueeze(0)), (player.hx, player.cx)))
            R = value.data
        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                R = R.cuda()
        # add V(s_{last+1}) to player.values
        player.values.append(Variable(R))
        policy_loss = 0 #loss of Actor
        value_loss = 0 #loss of Critic
        gae = torch.zeros(1, 1) #gae=[[0]]
        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                gae = gae.cuda()
        R = Variable(R)
        # player.action_train()-->get player.rewards
        for i in reversed(range(len(player.rewards))):
            R = args.gamma * R + player.rewards[i] # gamma*V_{t+1} + r_{t}
            advantage = R - player.values[i] #TD-error = gamma*V_{t+1} + r_{t} - V_{t}
            # loss function of critic
            value_loss = value_loss + 0.5 * advantage.pow(2) # sum_t(0.5*(TD-error)^2)

            # Generalized Advantage Estimation (GAE)
            # delta_t: TD-error = gamma*V_{t+1} + r_{t} - V_{t}
            # GAE
            delta_t = player.rewards[i] + args.gamma * player.values[i + 1].data - player.values[i].data
            gae = gae * args.gamma * args.tau + delta_t # tau: lambda in GAE
            
            # *******************************************************************************
            # loss function of actor
            # 技術：GAE + Entropy Regularization
            # gradient ascent: log_P*Advantage越大越好, entropy越大越好(讓輸出機率不會過於集中)
            # 因為是做gradient descent，所以以上兩者都要加負號。
            # *******************************************************************************
            policy_loss = policy_loss - player.log_probs[i] * Variable(gae) - 0.01 * player.entropies[i]
        # reset the gradient of worker
        player.model.zero_grad()
        # policy(actor) & value(critic): backward()-->calculate gradient for worker
        (policy_loss + 0.5 * value_loss).backward()
        # sent gradient from worker to shared model!
        ensure_shared_grads(player.model, shared_model, gpu=gpu_id >= 0) 
        # update shared model (because shared model's parameters are recorded in optimizer) 
        optimizer.step()
        # logger
#         log['{0}_train_{1}_log'.format(args.env, rank)].info(
#                     "Time {0}, policy loss {1}, episode length {2}".
#                     format(time.strftime("%Hh %Mm %Ss",time.gmtime(time.time() - start_time)),
#                            policy_loss.item(), len(player.rewards)))        
        player.clear_actions()
        
        if episode_passed > args.max_train_episode:
            break