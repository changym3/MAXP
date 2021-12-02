
import yaml
import os
import glob
from easydict import EasyDict

from srcs.lightning_model import MaxpLightning


def load_model(directory, series):
    ckpt_path_pattern = os.path.join('.', directory, series, 'checkpoints', '*.ckpt')
    ckpt_path = glob.glob(ckpt_path_pattern)[0]
    ckpt = MaxpLightning.load_from_checkpoint(ckpt_path)
    return ckpt


def dump_config(config, model, descriptions):
    obj = EasyDict(config)
    obj['descriptions'] = descriptions
    file_name = './configurations/{}.yaml'.format(model)
    with open(file_name, 'w') as f:
        yaml.dump(obj, f)
    print('Successfully dump config files to {}'.format(file_name))

def load_config(model='triangle', version=0):
    file_name = './configurations/{}.yaml'.format(model)
    with open(file_name, 'r') as f:
        config = yaml.load(f)
    config.pop('descriptions')
    return config


def get_light_config(series='attempts'):
    config = EasyDict()
    # dataset


    config.DATA_PATH = './official_data/processed_data/'
    config.NUM_FEATS = 300
    config.NUM_CLASSES = 23
    # architecture
    config.gnn_model = 'GAT'  
            # ['graphsage', 'graphconv', 'graphattn'] 
            # ['HetSAGE', 'HetGAT', 'HetLGC']
    
    config.hidden_dim = 256
    config.num_layers = 2
    config.fanout = [10] * config.num_layers
    # model 
    config.etypes = ['bicited']  # ['bicited']    ['cite', 'cited', 'bicited']    ['cite', 'cited'] 
    config.het_combine = 'attn' # 'attn', 'mean'
    config.num_heads = 4
    config.dropout = 0.3
    config.feat_drop = 0
    config.attn_drop = 0
    # training
    config.lr = 0.001
    config.optim = 'adam'   # 'adam', 'SGD'
    config.scheduler = 'cycle' # 'cycle', 'step', 'plateau', None
    config.l2_norm = 0.01

    config.num_epochs = 100
    config.batch_size = 1024 * 4
        # HetGAT with single edge can be 4096 * 32
    config.patience = 3
    # environments
    config.gpu_id = 1
    config.num_workers = 0
    # model info
    config.directory = 'light'
    config.series = series

    # print('############### Config info: ###############')
    # print(json.dumps(config, indent=4))
    return config


def get_cubic_config(series='attempts'):
    config = get_light_config(series)
    config.directory = 'cubic'
    config.batch_size = 4096 

    config.lr = 0.001
    config.hidden_dim = 128
    config.num_layers = 2
    config.fanout = [20] * config.num_layers
    config.etypes = ['cite', 'bicited'] 
    return config




def get_test_config():
    config = get_light_config()
    config.directory = 'logs'
    return config