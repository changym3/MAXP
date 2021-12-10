import argparse
import datetime as dt
import json
import tqdm
from easydict import EasyDict
import warnings
import torch
import pytorch_lightning as pl



from ..utils import load_default_split, load_graph, save_submission
from .gnn_lightning import MaxpDataModule, MaxpLightning
from ..utils import get_trainer, get_parser, graph_predict

# disable batch_size warning which comes from dgl input nodes
warnings.filterwarnings("ignore", message="Trying to infer the `batch_size` from an ambiguous collection.")



def main(config):
    print('----------------config----------------')
    print(json.dumps(vars(config), indent=4))
    print('--------------------------------------')

    start_time = dt.datetime.now()
    graph, node_feat, labels = load_graph(config.data_dir, config.etypes)
    tr_label_idx, val_label_idx, test_label_idx = load_default_split(config.data_dir)
    device = torch.device('cuda:{}'.format(config.gpu_id))
    datamodule = MaxpDataModule(config, (graph, node_feat, labels), (tr_label_idx, val_label_idx, test_label_idx), device)
    model = MaxpLightning(config, steps_per_epoch=len(datamodule.train_dataloader()))
    trainer = get_trainer(config)
    trainer.fit(model, datamodule=datamodule)
    end_time = dt.datetime.now()

    print('----------------Finished----------------')
    print('Finished at {}', dt.datetime.now().strftime('%m-%d %H:%M'))
    print('Using {} seconds'.format((end_time-start_time).seconds))
    print('The model is saved at', trainer.checkpoint_callback.best_model_path)
    print('The model performance of last epoch :', json.dumps(trainer.progress_bar_dict, indent=4))

    if config.training_mode == 'inference':
        nids, preds = graph_predict(model, datamodule.predict_dataloader(), device)
        save_submission(nids, preds, 
            filename='{}_{}'.format(config.name, config.version), 
            data_dir=config.data_dir, sub_dir=config.sub_dir
        )


if __name__ == '__main__':
    parser = get_parser()

    # args = parser.parse_args(''.split())    
    # config = EasyDict(vars(args))

    args = parser.parse_args()
    config = EasyDict(vars(args))
    
    main(config)