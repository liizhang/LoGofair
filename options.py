import argparse
from config import *

def arg_parser():
    parser = argparse.ArgumentParser()

    # Algoritha
    parser.add_argument('--algorithm', help = 'name of algorithm',
                        type = str, choices = OPTIMIZERS, default = 'logofair')
    parser.add_argument('--num_users', help = 'number of users', 
                        type = int, default = 5)
    
    # data set
    parser.add_argument('--data', help = 'name of dataset',
                        type = str, default = 'adult')
    
    parser.add_argument('--alpha', help = 'dirichlet parameter for dataset partition. A smaller alpha indicates a higher degree of heterogeneity in the federated dataset.',
                        type = float, default = 0.5)

    parser.add_argument('--data_setting', help = 'operations on raw datas to generate decentralized data set; input a dict',
                        type = dict, default = {'sensitive_attr': 'sex', 'dirichlet':True, 'by sensitive':True, 'alpha': 0.5, 'generate': False})
    
    parser.add_argument('--seed', help = 'operations on raw datas to generate decentralized data set; input a dict',
                        type = int, default = 111)
    
    # model
    parser.add_argument('--model', help = 'name of local model;',
                        type = str, choices = MODELS, default = 'logistic')
    parser.add_argument('--criterion', help = 'name of loss function',
                        type = str, choices = CRITERIA, default = 'bceloss')
    parser.add_argument('--wd', help = 'weight decay parameter;',
                        type = float, default = 1e-4)
    parser.add_argument('--gpu', help = 'use gpu (default: True)',
                        default = True, action = 'store_true')

    # federated arguments
    parser.add_argument('--server', help = 'type of server',
                        type = str, default = 'server', choices = SERVERS)
    parser.add_argument('--num_round', help = 'number of rounds to simulate',
                        type = int, default = 25)
    parser.add_argument('--eval_round', help = 'evaluate every ___ rounds',
                        type = int, default = 1)
    parser.add_argument('--clients_per_round', help = 'number of clients selected per round',
                        type = int, default = 10)
    parser.add_argument('--batch_size', help = 'batch size when clients train on data',
                        type = int, default = 512)
    parser.add_argument('--num_epoch', help = 'number of rounds for solving the personalization sub-problem when clients train on data',
                        type = int, default = 1)
    parser.add_argument('--num_local_round', help = 'number of local rounds for local update',
                        type = int, default = 40)
    parser.add_argument('--local_lr', help = 'learning rate for local update',
                        type = float, default = 0.005)
    parser.add_argument('--local_optimizer', help = 'optimizer for local training',
                        type = str, default = 'adam')
    parser.add_argument('--test_local', help = 'if test model with local params',
                        type = bool, default = False)
    parser.add_argument('--print_result', help = 'if print the result',
                        type = bool, default = True)
    ## Fairness
    parser.add_argument('--fairness_measure', help = 'fairness measure: DP or EO',
                        type = list, default = ['DP'])
    
    # arguments for federated algorithm

    ## FedAvg
    parser.add_argument('--aggr', help = 'aggregation method',
                        type = str, default = 'mean')
    parser.add_argument('--simple_average', help = 'if simple average used',
                        type = str, default = True)
    parser.add_argument('--weight_aggr', help = 'weighted aggregation',
                        type = float, default = False)
    parser.add_argument('--beta', help = 'model params momentum',
                        type = float, default = 0.95)
    
    ## FedFB
    parser.add_argument('--FB_alpha', help = 'learning rate for dynamic weight',
                        type = float, default = 0.3)
    
    ## AFL
    parser.add_argument('--weight_lr', help = 'learning rate for dynamic weight',
                        type = float, default = 0.5)
    
    ##FairFed
    parser.add_argument('--fairfed_beta', help = 'parameter to control the aggregation',
                        type = float, default = 0.5)
    parser.add_argument('--fair_measure', help = 'use which fair measure',
                        type = str, default = "DP")
    
    ## FPP
    parser.add_argument('--fairness_metric', help = 'fairness_metric',
                        type = str, default = 'DP')
    parser.add_argument('--fairness_constraints', help = 'fairness constraints',
                        type = dict, default = {'global':0.02, 'local': 0.02})
    parser.add_argument('--FFP_beta', help = 'beta',
                        type = float, default = 1000)
    parser.add_argument('--post_local_round_mu', help = 'local round',
                        type = int, default = 20)
    parser.add_argument('--post_local_round_lamb', help = 'local round',
                        type = int, default = 20)
    parser.add_argument('--post_round', help = 'post local round',
                        type = int, default = 30)
    parser.add_argument('--post_lr', help = 'lr',
                        type = float, default = 0.005)
    parser.add_argument('--calibration', help = 'calibration',
                        type = float, default = True)
    parser.add_argument('--fair_save', help = 'save the result',
                        type = float, default = True)
    
    
    args = parser.parse_args()

    args.fairness_constraints['fairness_measure'] = args.fairness_metric
    args.fairness_constraints['fairness_measure'] = args.fairness_metric
    args.data_setting['alpha'] = args.alpha
    return args