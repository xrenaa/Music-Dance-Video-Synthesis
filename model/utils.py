"""
Tensorboard logger code referenced from:
https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/04-utils/
Other helper functions:
https://github.com/cs230-stanford/cs230-stanford.github.io
"""

import json
import logging
import os
import shutil
import torch
import openpyxl
import argparse
import numpy as np
from collections import OrderedDict

try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO  # Python 3.x


class Params():
    """Class that loads hyper-parameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


class RunningAverage():
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


def set_logger(log_file, logger_name = None):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger(logger_name)# logger name, can have many handlers
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s:%(levelname)s: %(message)s')

    if not logger.handlers: # logger.handlers=[], no handlers
        # Logging to a file
        file_handler = logging.FileHandler(log_file,mode='w')# log file
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def save_checkpoint(state, epoch,is_best,save_best_ever_n_epoch,checkpointpath,start_epoch):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'

    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    checkpoint = checkpointpath
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    # else:
        # print("Checkpoint Directory exists! ")
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    torch.save(state, filepath)

    parent_dir = os.path.dirname(checkpoint)
    file_name1 = os.path.join(parent_dir, 'best_accuracy_series.xlsx')
    if epoch == start_epoch+1:
        wb = openpyxl.Workbook(file_name1)
        wb.save(filename=file_name1)
        wb.close()

    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))

    if epoch % save_best_ever_n_epoch == 0 :
        shutil.copyfile(os.path.join(checkpoint, 'best.pth.tar'),os.path.join(checkpoint,'epoch{}.pth.tar'.format(str(epoch))))

        wb = openpyxl.load_workbook(file_name1)
        ws = wb['Sheet']
        max_column = ws.max_column
        ws.cell(row=1, column=max_column+1).value = 'epoch{}'.format(str(epoch))
        ws.cell(row=2, column=max_column+1).value = (1-state['best_val_acc'])*100
        wb.save(filename=file_name1)
        wb.close()

def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.

    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise ("File doesn't exist {}".format(checkpoint))
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint)
    else:
        # this helps avoid errors when loading single-GPU-trained weights onto CPU-model
        checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)

    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint


# initial way
# use the next 3 functions to initial a model, eg at the third function
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        if m.bias is not None:
            m.bias.data.fill_(0)

    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        if m.bias is not None:
            m.bias.data.fill_(0)

    elif classname.find('LSTM') != -1:
        for name, param in m.named_parameters():
            if 'bias' in name:
                torch.nn.init.constant(param, 0.0)
            elif 'weight' in name:
                torch.nn.init.orthogonal(param)

        # Initialize biases for LSTM’s forget gate to 1 to remember more by default. Similarly, initialize biases for GRU’s reset gate to -1.
        for names in m._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(m, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.)

    elif classname.find('GRU') != -1:
        for name, param in m.named_parameters():
            if 'bias' in name:
                torch.nn.init.constant(param, 0.0)
            elif 'weight' in name:
                torch.nn.init.orthogonal(param)


def initial_model_weight(layers):
    for layer in layers:
        if list(layer.children()) == []:
            weights_init(layer)
            # print('weight initial finished!')
        else:
            for sub_layer in list(layer.children()):
                initial_model_weight([sub_layer])

# initial weight
# use this when define models
# utils.initial_model_weight(layers = list(self.children()))
# print('weight initial finished!')


# str2bool for argparse; argparse cannot use boolean directly
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')