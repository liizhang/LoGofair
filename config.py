# Global parameters
DATASETS = ['mnist', 'synthetic', 'emnist', 'fmnist', 'cifar']

MODELS = ['logistic', '2nn', '1nn', 'cnn']

ALGORITHMS_MAPPING = {'fedavg': 'FedAvg',
                         'afl': 'AFL',
                     'fairfed':'FairFed',
                       'lfb':'LFB',
                       'localfb':'LocalFB',
                       'fedfb':'FedFB',
                       'localada':'Localadapt',
                       'logofair':"FedFairPost"}

OPTIMIZERS = ALGORITHMS_MAPPING.keys()

CRITERIA = ['celoss', 'mseloss']

ATTACKS = ['same_value', 'sign_flip', 'gaussian', 'data_poison']

SERVERTYPE = {'server': 'Server',
                'robust_server': 'RobustServer',
                'server_sketch': 'ServerSketch',
                'server_lg': 'ServerLg',
                'server_local': 'ServerLocal',
                'server_lbgm': 'ServerLBGM',
                'server_gradient': 'ServerGradient'}
SERVERS = SERVERTYPE.keys()

AGGR = ['mean', 'median', 'krum']