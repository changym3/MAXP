
import argparse
import json
import os
import yaml
import glob
from easydict import EasyDict
import torch

from .gnn_model import MaxpLightning
from srcs.utils.data import GraphDataset, GraphDataModule, Prediction


def main(args):
    print('----------------config----------------')
    print(json.dumps(vars(args), indent=4))
    print('--------------------------------------')

    model_dir = args.model_dir
    model_param_path = os.path.join(model_dir, 'hparams.yaml')
    model_ckpt_path = glob.glob(os.path.join(model_dir, 'checkpoints', '*.ckpt'))[0]
    with open(model_param_path, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
        config = EasyDict(params)

    dataset = GraphDataset(config.data_dir, config.etypes)
    device = torch.device('cuda:{}'.format(args.gpu_id))
    datamodule = GraphDataModule(config, dataset, device)

    model = MaxpLightning(config, steps_per_epoch=len(datamodule.train_dataloader()))
    model = MaxpLightning.load_from_checkpoint(model_ckpt_path, steps_per_epoch=len(datamodule.train_dataloader()))

    nids, preds, labels = model.predict(datamodule.all_nid_dataloader(), device)
    teacher = Prediction(nids, preds, labels)
    model_pred_path = os.path.join(model_dir, 'prediction.pkl')
    teacher.save(model_pred_path)
    print('Saving predictions finished')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='/home/changym/competitions/MAXP/logs/qbase_v2/base/')
    parser.add_argument('--gpu_id', type=int, default=0)
    args = parser.parse_args()
    print(1)
    main(args)


# python -m srcs.GNN.gnn_save_pred --gpu_id 0