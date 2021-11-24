import torch
import torch.nn.functional as F
from torch.autograd import Variable
'''
player = Agent(None, env, args, None)
player.gpu_id = gpu_id
player.model = A3Clstm(player.env.observation_space.shape[0],
                       player.env.action_space)

player.state = player.env.reset()
player.state = torch.from_numpy(player.state).float()
'''
class Agent(object):
    def __init__(self, model, env, args, state):
        self.model = model
        self.env = env
        self.state = state
        self.hx = None
        self.cx = None
        self.eps_len = 0
        self.args = args
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.done = True
        self.info = None
        self.reward = 0
        self.gpu_id = -1

    def action_train(self):
        '''
        model.forward() need a input that data type is "tuple(state, (hx, cx))"
            -->return：critic_linear(x), actor_linear(x), (hx, cx)
        '''
        value, logit, (self.hx, self.cx) = self.model((Variable(self.state.unsqueeze(0)), (self.hx, self.cx)))
        '''
        logit = actor_linear(x)
        因為actor_linear(x)當初設計時不是回傳機率，故添加softmax(dim=1)與log_softmax(dim=1)
        '''
        prob = F.softmax(logit, dim=1)
        log_prob = F.log_softmax(logit, dim=1)
        # Entropy = -logP*logP
        entropy = -(log_prob * prob).sum(1)
        self.entropies.append(entropy)
        # prob.multinomial(1): 依照prob機率大小抽出1個index
        action = prob.multinomial(1).data
        # gather(dim=1,shape=(batch size, 1)之行向量) --> (0,0) (1,0) (2,0) (3,0) ... 
        # --> 替換成action之index --> (0,idx0) (1,idx1) (2,idx2) (3,idx3)
        # --> 輸出log_prob --> shape=((batch size, 1))
        log_prob = log_prob.gather(1, Variable(action))
        # step
        state, self.reward, self.done, self.info = self.env.step(action.cpu().numpy())
        # state轉成torch.FloatTensor()
        self.state = torch.from_numpy(state).float()
        if self.gpu_id >= 0:
            # 把state丟給特定的gpu
            with torch.cuda.device(self.gpu_id):
                self.state = self.state.cuda() 
        # 控制reward為 -1 | 1
#         self.reward = max(min(self.reward, 1), -1)
        if self.done:
            self.reward = -10
        self.values.append(value) # type(value) = Tensor()
        self.log_probs.append(log_prob)
        self.rewards.append(self.reward)
        return self

    def action_test(self):
        with torch.no_grad():
            if self.done: #初始時done=True
                if self.gpu_id >= 0:
                    with torch.cuda.device(self.gpu_id):
                        #把 cx, hx丟進去特定的cuda，初始的cx, hx為0 vector
                        self.cx = Variable(
                            torch.zeros(1, 512).cuda())
                        self.hx = Variable(
                            torch.zeros(1, 512).cuda())
                else:
                    self.cx = Variable(torch.zeros(1, 512))
                    self.hx = Variable(torch.zeros(1, 512))
            else: #遊戲開始後，直到結束前done=False
                self.cx = Variable(self.cx.data)
                self.hx = Variable(self.hx.data)
            # forward
            value, logit, (self.hx, self.cx) = self.model((Variable(
                self.state.unsqueeze(0)), (self.hx, self.cx)))
        prob = F.softmax(logit, dim=1)
        #action = prob.max(1)[1].data.cpu().numpy() #(1):dim=1 #[1]:index
        action = prob.multinomial(1).data.cpu().numpy()
        # action: array([1], dtype=int64)
        state, self.reward, self.done, self.info = self.env.step(action[0])
        self.state = torch.from_numpy(state).float()
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.state = self.state.cuda()
        self.eps_len += 1
        return self

    def clear_actions(self):
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        return self