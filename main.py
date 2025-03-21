import numpy as np
import argparse
import importlib
import torch
import os

from options import arg_parser
from fedlearn.utils.data_utils import get_data
from config import ALGORITHMS_MAPPING

def read_options():

    # read options
    args = arg_parser()

    # Set options as a dict
    try: 
        options = vars(args)
    except IOError as msg:
        args.error(str(msg))
    print(options)

    # use gpu
    if options['gpu']:
        if torch.cuda.is_available():
            options['device'] = torch.device("cuda:0")
        else:
            options['gpu'] = False
            print('gpu is unavailable')

    # Set seeds
    np.random.seed(1 + options['seed'])
    torch.manual_seed(12 + options['seed'])
    if options['gpu']:
        torch.cuda.manual_seed_all(123 + options['seed'])


    # Print arguments and return
    max_length = max([len(key) for key in options.keys()])
    fmt_string = '\t%' + str(max_length) + 's : %s'
    print('>>> Arguments:')
    for keyPair in sorted(options.items()):
        print(fmt_string % keyPair)

    # Load selected server
    server_path = 'fedlearn.algorithm.%s' % ALGORITHMS_MAPPING[options['algorithm'].lower()]
    mod = importlib.import_module(server_path)
    server_class = getattr(mod, 'Server')

    return options, server_class

def main():
    # Parse command line arguments
    options, server_class = read_options()

    # `dataset` : (cids, train_data, valid_data, test_data)
    data_info = get_data(options)
    selected_server = server_class(data_info, options)
    selected_server.train()

if __name__ == '__main__':
    main()