import argparse
import datetime as dt
import json
import tqdm
from easydict import EasyDict
import warnings
import torch
import pytorch_lightning as pl



from ..utils import load_graph, save_submission
from .gnn_lightning import MaxpDataModule, MaxpLightning, graph_predict
from ..model_utils import get_trainer, get_parser

# disable batch_size warning which comes from dgl input nodes
warnings.filterwarnings("ignore", message="Trying to infer the `batch_size` from an ambiguous collection.")



def main(config):
    print('----------------config----------------')
    print(json.dumps(vars(config), indent=4))
    print('--------------------------------------')

    start_time = dt.datetime.now()
    device = torch.device('cuda:{}'.format(config.gpu_id))
    datamodule = MaxpDataModule(config, device)
    model = MaxpLightning(config)
    trainer = get_trainer(config)
    trainer.fit(model, datamodule=datamodule)
    end_time = dt.datetime.now()

    print('----------------Finished----------------')
    print('Finished at {}', dt.datetime.now().strftime('%m-%d %H:%M'))
    print('Using {} seconds'.format((end_time-start_time).seconds))
    print('The model is saved at', trainer.checkpoint_callback.best_model_path)
    print('The model performance of last epoch :', json.dumps(trainer.progress_bar_dict, indent=4))

    if config.inference:
        nids, preds = graph_predict(model, datamodule, device)
        save_submission(nids, preds, 
            filename='{}_{}'.format(config.name, config.version), 
            data_dir=config.data_dir, sub_dir=config.sub_dir
        )


if __name__ == '__main__':
    parser = get_parser()

    # args = parser.parse_args(''.split())    
    # config = EasyDict(vars(args))

    args = parser.parse_args()
    args.fanouts = [args.fanout] * args.num_layers
    config = EasyDict(vars(args))
    
    main(config)