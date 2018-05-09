import yaml
import numpy as np
from pprint import pprint
from easydict import EasyDict as edict


class config(object):
    def __init__(self, filename=''):
        self.config_dict = edict()
        # network training
        self.config_dict.TRAIN = edict()
        self.config_dict.TRAIN.EPOCHS = 10
        self.config_dict.TRAIN.START_EPOCH = 0
        self.config_dict.TRAIN.BATCH_SIZE = 16
        self.config_dict.TRAIN.INIT_LEARNING_RATE = 0.01
        self.config_dict.TRAIN.MOMENTUM = 0.9
        self.config_dict.TRAIN.NESTEROV = True
        self.config_dict.TRAIN.WEIGHT_DECAY = 5e-4

        # network architecture
        self.config_dict.NET = edict()
        self.config_dict.NET.DEPTH = 5
        self.config_dict.NET.NUM_OFFSETS = 15
        self.config_dict.NET.NUM_CLASSES = 2

        # image
        self.config_dict.IMAGE = edict()
        self.config_dict.IMAGE.HEIGHT = 128
        self.config_dict.IMAGE.WIDTH = 128
        self.config_dict.IMAGE.CHANNELS = 3

        # I/O path
        self.config_dict.PATH = edict()
        self.config_dict.PATH.EXP_NAME = 'unet_5'
        self.config_dict.PATH.TRAIN_DIR = 'data/train_val/split0.9_seed0'
        self.config_dict.PATH.TEST_DIR = 'data/test'

        # log
        self.config_dict.TENSORBOARD = False

        # load from yaml
        if filename is not '':
            print('Load configuration from {}:'.format(filename))
            self.load_from_yaml(filename)
        else:
            print('Use default configuration:')
        pprint(self.config_dict)

    def merge_edict(self, a, b):
        """Merge config dictionary a into config dictionary b, clobbering the
        options in b whenever they are also specified in a.
        """
        if type(a) is not edict:
            return

        for k, v in a.items():
            # a must specify keys that are in b
            if k not in b:
                raise KeyError('{} is not a valid config key'.format(k))

            # the types must match, too
            old_type = type(b[k])
            if old_type is not type(v):
                if isinstance(b[k], np.ndarray):
                    v = np.array(v, dtype=b[k].dtype)
                else:
                    raise ValueError(('Type mismatch ({} vs. {}) '
                                      'for config key: {}').format(type(b[k]),
                                                                   type(v), k))

            # recursively merge dicts for nested edict
            if type(v) is edict:
                try:
                    self.merge_edict(a[k], b[k])
                except:
                    print('Error under config key: {}'.format(k))
                    raise
            else:
                b[k] = v

    def load_from_yaml(self, filename):
        with open(filename, 'r') as f:
            yaml_cfg = edict(yaml.load(f))
        self.merge_edict(yaml_cfg, self.config_dict)


if __name__ == '__main__':
    config_test = config('example.yml')
