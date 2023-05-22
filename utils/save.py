import os
import json
import sys
import yaml
import copy
import shutil
import torch

def ensure_dir(d, verbose=True):
    if not os.path.exists(d):
        if verbose:
            print("Directory {} do not exist; creating...".format(d))
        os.makedirs(d)

def save_config(args, path, verbose=True):
    config = copy.deepcopy(args)
    config = vars(config)
    for key,value in config.items():
        if isinstance (value,str):
            continue
        else:
            config[key] = str(value)
    with open(path, 'w') as outfile:
        json.dump(config, outfile, indent=2)
    if verbose:
        print("Config saved to file {}".format(path))
    return config

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')
        
class FileLogger(object):
    """
    A file logger that opens the file periodically and write to it.
    """

    def __init__(self, filename, header=None):
        self.filename = filename
        if os.path.exists(filename):
            # remove the old file
            os.remove(filename)
        if header is not None:
            with open(filename, 'w') as out:
                print(header, file=out)

    def log(self, message):
        # print(message)
        with open(self.filename, 'a') as out:
            print(message, file=out)