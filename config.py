import argparse
import logging
import os
import random
import sys
import time
from datetime import datetime

import torch


def get_config():
    parser = argparse.ArgumentParser()
    num_classes = {'sst2': 2, 'sst5': 5, 'cr': 2, 'subj': 2, 'mr': 2, 'trec': 6, 'mpqa': 2, 'ie': 3}
    max_lengths = {'sst2': 53, 'sst5': 53, 'cr': 100, 'subj': 108, 'mr': 53, 'trec': 33, 'mpqa': 34, 'ie': 75}
    query_lengths = {'sst2': 14, 'sst5': 20, 'cr': 14, 'subj': 14, 'mr': 14, 'trec': 22, 'mpqa': 14, 'ie': 16}

    '''Base'''
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--dataset', type=str, default='ie', choices=num_classes.keys())
    parser.add_argument('--max_lengths', type=str, default='ie', choices=max_lengths.keys())
    parser.add_argument('--query_lengths', type=str, default='ie', choices=query_lengths.keys())
    parser.add_argument('--model_name', type=str, default='bert',
                        choices=['bert', 'roberta', 'roberta-large', 'wsp-base', 'wsp-large'])
    parser.add_argument('--method_name', type=str, default='san',
                        choices=['cls', 'cls_extend_lstm', 'cls_extend_bilstm', 'label', 'text_last_hidden',
                                 'text_hiddens', 'cnn+rnn', 'cls_explain', 'self_attention', 'san'])

    '''Optimization'''
    parser.add_argument('--train_batch_size', type=int, default=1)
    parser.add_argument('--test_batch_size', type=int, default=64)
    parser.add_argument('--num_epoch', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--decay', type=float, default=0.01)
    parser.add_argument('--eps', type=float, default=1e-8)
    parser.add_argument('--prompt_lengths',type=int,default=5)

    '''Environment'''
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--backend', default=False, action='store_true')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--timestamp', type=int, default='{:.0f}{:03}'.format(time.time(), random.randint(0, 999)))

    args = parser.parse_args()
    args.num_classes = num_classes[args.dataset]
    args.max_lengths = max_lengths[args.max_lengths]
    args.query_lengths = query_lengths[args.query_lengths]
    args.device = torch.device(args.device)

    '''logger'''
    args.log_name = '{}_{}_{}_{}.log'.format(args.model_name, args.method_name, args.dataset,
                                             datetime.now().strftime('%Y-%m-%d_%H-%M-%S')[2:])
    if not os.path.exists('logs'):
        os.mkdir('logs')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.addHandler(logging.FileHandler(os.path.join('logs', args.log_name)))
    return args, logger
