
import yaml
import os
import glob
from easydict import EasyDict

from srcs.lightning_model import MaxpLightning


def load_model(directory, series, version):
    ckpt_path_pattern = os.path.join('.', directory, series, version, 'checkpoints', '*.ckpt')
    ckpt_path = glob.glob(ckpt_path_pattern)[0]
    ckpt = MaxpLightning.load_from_checkpoint(ckpt_path)
    return ckpt


def dump_config(config, model, version, descriptions):
    obj = EasyDict(config)
    obj['descriptions'] = descriptions
    file_name = './configurations/{}_v{}.yaml'.format(model, version)
    with open(file_name, 'w') as f:
        yaml.dump(obj, f)
    print('Successfully dump config files to {}'.format(file_name))

def load_config(model='triangle', version=0):
    file_name = './configurations/{}_v{}.yaml'.format(model, version)
    with open(file_name, 'r') as f:
        config = yaml.load(f)
    config.pop('descriptions')
    return config


def get_light_config(series='test', version=None):
    config = EasyDict()
    # dataset
    config['DATA_PATH'] = './official_data/processed_data/'
    config['NUM_FEATS'] = 300
    config['NUM_CLASSES'] = 23
    # architecture
    config['gnn_model'] = 'HetGAT'  
            # ['graphsage', 'graphconv', 'graphattn'] 
            # ['HetSAGE', 'HetGAT', 'HetLGC']
    
    config['hidden_dim'] = 64
    config['num_layers'] = 2
    config['fanout'] = [10] * config.num_layers
    # model 
    config['etypes'] = ['bicited']  # ['bicited']    ['cite', 'cited', 'bicited']    ['cite', 'cited'] 
    config['het_combine'] = 'attn' # 'attn', 'mean'
    config['num_heads'] = 4
    config['feat_drop'] = 0.3
    config['attn_drop'] = 0.3
    # training
    config['lr'] = 0.005
    config['num_epochs'] = 1000
    config['batch_size'] = 4096 * 32
        # HetGAT with single edge can be 4096 * 32
    config['patience'] = 10
    # environments
    config['gpu_id'] = 1
    config['num_workers'] = 0
    # model info
    config['directory'] = 'light'
    config['series'] = series
    config['version'] = version

    # print('############### Config info: ###############')
    # print(json.dumps(config, indent=4))
    return config


def get_cubic_config(series='test', version=None):
    config = EasyDict()
    # dataset
    config['DATA_PATH'] = './official_data/processed_data/'
    config['NUM_FEATS'] = 300
    config['NUM_CLASSES'] = 23
    # architecture
    config['gnn_model'] = 'HetGAT'  
            # ['graphsage', 'graphconv', 'graphattn'] 
            # ['HetSAGE', 'HetGAT', 'HetLGC']

    config['hidden_dim'] = 128
    config['num_layers'] = 2
    config['fanout'] = [20] * config.num_layers
    # model 
    config['etypes'] = ['bicited']  # ['bicited']    ['cite', 'cited', 'bicited']    ['cite', 'cited'] 
    config['het_combine'] = 'attn' # 'attn', 'mean'
    config['num_heads'] = 4
    config['feat_drop'] = 0.3
    config['attn_drop'] = 0.3
    # training
    config['lr'] = 0.001
    config['num_epochs'] = 1000
    config['batch_size'] = 4096 * 16
        # sage can be 4096 * 32
    config['patience'] = 10
    # environments
    config['gpu_id'] = 1
    config['num_workers'] = 0
    # model info
    config['directory'] = 'cubic'
    config['series'] = series
    config['version'] = version

    # print('############### Config info: ###############')
    # print(json.dumps(config, indent=4))
    return config




def get_test_config():
    config = get_light_config()
    # model info
    config['directory'] = 'logs'
    config['series'] = 'test'
    config['version'] = None

    # print('############### Config info: ###############')
    # print(json.dumps(config, indent=4))
    return config