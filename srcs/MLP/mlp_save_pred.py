import os
import argparse
import yaml
import glob
from easydict import EasyDict
import torch

from srcs.utils.data import GraphDataset, NodeIDDataModule, Prediction
from .mlp_model import MLPModel


def main(args):
    model_dir = args.model_dir
    model_param_path = os.path.join(model_dir, 'hparams.yaml')
    model_ckpt_path = glob.glob(os.path.join(model_dir, 'checkpoints', '*.ckpt'))[0]
    with open(model_param_path, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
        config = EasyDict(params)

    dataset = GraphDataset(config.data_dir, config.etypes, config.use_degrees)
    datamodule = NodeIDDataModule(config, dataset)
    device = torch.device('cuda:{}'.format(args.gpu_id))

    model = MLPModel.load_from_checkpoint(model_ckpt_path, dataset=dataset, device=device, steps_per_epoch=len(datamodule.train_dataloader()))

    nids, preds, labels = model.predict(datamodule.all_nid_dataloader(), device)
    teacher = Prediction(nids, preds, labels)
    model_pred_path = os.path.join(model_dir, 'prediction.pkl')
    teacher.save(model_pred_path)
    print('Saving predictions finished')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='/home/changym/competitions/MAXP/logs/MLP_v1/base')
    parser.add_argument('--gpu_id', type=int, default=0)
    args = parser.parse_args()
    main(args)

# python -m srcs.MLP.mlp_save_pred 