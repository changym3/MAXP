import argparse
import datetime as dt
import json
import tqdm
from easydict import EasyDict
import warnings
import torch
import pytorch_lightning as pl


from .gnn_model import MaxpLightning
from srcs.utils.data import GraphDataModule, GraphDataset
from ..utils.utils import get_trainer, get_parser, graph_predict, save_submission

# disable batch_size warning which comes from dgl input nodes
warnings.filterwarnings("ignore", message="Trying to infer the `batch_size` from an ambiguous collection.")



def main(config):
    print('----------------config----------------')
    print(json.dumps(vars(config), indent=4))
    print('--------------------------------------')

    start_time = dt.datetime.now()
    dataset = GraphDataset(config.data_dir, config.etypes, config.use_degrees)
    device = torch.device('cuda:{}'.format(config.gpu_id))
    datamodule = GraphDataModule(config, dataset, device)    
    
    model = MaxpLightning(config, steps_per_epoch=len(datamodule.train_dataloader()))
    trainer = get_trainer(config)
    trainer.fit(model, datamodule=datamodule)
    end_time = dt.datetime.now()

    print('----------------Finished----------------')
    print('Finished at {}', dt.datetime.now().strftime('%m-%d %H:%M'))
    print('Using {} seconds'.format((end_time-start_time).seconds))
    print('The model is saved at', trainer.checkpoint_callback.best_model_path)
    print('The model performance of last epoch :', json.dumps(trainer.progress_bar_dict, indent=4))


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    config = EasyDict(vars(args))
    
    main(config)

# python -m srcs.GNN.gnn_main --num_layers 2 --fanout 10 10 --gpu_id 0