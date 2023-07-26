import argparse
import os.path as osp
import json
import torch
import transformers
import pandas as pd
import numpy as np
from tqdm import tqdm
import data_loader.data_loader as module_data
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from model.model import sim_matrix, compute_similarity
from sacred import Experiment
from utils.util import state_dict_data_parallel_fix
from trainer.trainer import verbose
ex = Experiment('test')

@ex.main
def run():

    # setup data_loader instances
    config._config['data_loader']['args']['split'] = 'test'
    config._config['data_loader']['args']['shuffle'] = False
    config._config['data_loader']['args']['sliding_window_stride'] = config._config['sliding_window_stride']
    config._config['data_loader']['args']['metadata_filename'] = args.metadata_filename
    config._config['data_loader']['args']['quva_dir'] = args.quva_dir
    config._config['data_loader']['args']['something_something_dir'] = args.something_something_dir
    config._config['data_loader']['args']['youtube_dir'] = args.youtube_dir
    config._config['data_loader']['args']['proficiency'] = args.proficiency
    data_loader = config.initialize('data_loader', module_data)
    tokenizer = transformers.AutoTokenizer.from_pretrained(config['arch']['args']['text_params']['model'])

    # build model architecture
    model = config.initialize('arch', module_arch)
    # get function handles of loss and metrics
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    #logger.info('Loading checkpoint: {} ...'.format(config.resume))

    if config.resume is not None:
        checkpoint = torch.load(config.resume)
        state_dict = checkpoint['state_dict']
        new_state_dict = state_dict_data_parallel_fix(state_dict, model.state_dict())
        model.load_state_dict(new_state_dict, strict=True)
    else:
        print('Using random weights')

    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    results = dict()
    num_correct = num_examples = 0
    print(len(data_loader))
    with torch.no_grad():
        for i, data in tqdm(tqdm(enumerate(data_loader))):
            text_inputs = tokenizer(
                data['text'],
                return_tensors='pt',
                padding=True,
                truncation=True,
            ).to('cuda:0')
            data['video'] = data['video'].to(device)
            inputs = {
                'video': data['video'],
                'text': text_inputs,
            }
            text_embeds, vid_embeds = model(inputs)
            output, _ = compute_similarity(text_embeds, vid_embeds)
            batch_size, offset = vid_embeds.shape[0], 0
            for i in range(batch_size):
                num_text = data['num_text'][i]
                this = output[offset:offset+num_text, i]
                results[data['key'][i]] = {'scores': this.tolist()}
                offset += num_text

    with open(config._config['output_file'], 'w') as f:
        json.dump(results, f, indent=4)
    print('done.')


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')

    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-s', '--sliding_window_stride', default=-1, type=int,
                      help='test time temporal augmentation, repeat samples with different start times.')
    args.add_argument('--metadata_filename', default='relations.json',
                      help='annotations file name (e.g. relations.json)')
    args.add_argument('--quva_dir', default=None,
                      help='full path to the QUVA dataset root dir.')
    args.add_argument('--something_something_dir', default=None,
                      help='full path to the something something dataset (v2) video dir.')
    args.add_argument('--youtube_dir', default=None,
                      help='full path to the youtube download dir.')
    args.add_argument('-o', '--output_file', default=None, required=True)
    args.add_argument("--proficiency", action="store_true",
                      help="use the profiency task captions.")

    config = ConfigParser(args, test=True)
    # hack to get sliding into config
    args = args.parse_args()
    config._config['sliding_window_stride'] = args.sliding_window_stride
    config._config['output_file'] = osp.abspath(osp.expanduser(args.output_file))
    ex.add_config(config.config)

    ex.run()
