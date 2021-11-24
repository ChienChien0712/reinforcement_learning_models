import numpy as np
import torch
import json
import logging

def setup_logger(logger_name, log_file, level=logging.INFO):
    # 把特定的logger_name設定好，不用回傳。
        # l <- logging.getLogger(logger_name)
        #1. l <- setLevel <- level <- logging.INFO
        #2. l <- addHandler <- fileHandler <- logging.FileHandler(log_path, mode='w')
        #3. l <- addHandler <- streamHandler <- logging.StreamHandler()
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)
    l.addHandler(streamHandler)


def read_config(file_path):
    """Read JSON config."""
    json_object = json.load(open(file_path, 'r'))
    return json_object


def norm_col_init(weights, std=1.0):
    '''
    initialization method:
        init_w ~ ND(mean,std)
            where mean=0, std = gain * sqrt(2/fan_in + fan_out)
            
    other methods:
        torch.nn.init.xavier_normal_(tensor, gain=1)
    '''
    x = torch.randn(weights.size())
    x *= std / torch.sqrt((x**2).sum(1, keepdim=True))
    return x


def ensure_shared_grads(model, shared_model, gpu=False):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None and not gpu:
            return
        elif not gpu:
            shared_param._grad = param.grad
        else:
            shared_param._grad = param.grad.cpu()


def weights_init(m):
    # m: model
    classname = m.__class__.__name__
    '''
    m.__class__.__name__: 'Conv2d' | 'Linear' | 'MaxPool2d' | 'LSTMCell'
    
    initialization method:
        init_w ~ U(−a,a), where a = gain * sqrt(6/fan_in+fan_out)
    
        setting of 'gain' is based on the selected activate funtion.
    
    other methods:
        relu_gain = nn.init.calculate_gain('relu')
        torch.nn.init.xavier_uniform_(tensor, gain=relu_gain)

    '''
    if classname.find('Conv') != -1: 
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)