import copy
import torch
import random
import json
import numpy as np
import pandas as pd
import os
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from time import localtime, strftime
import time

import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms
from pathlib import Path
from PIL import Image
import multiprocessing
import threading
import concurrent.futures

from fedlearn.utils.sampling import *
# from data.celeba.metadata_to_json import celeba_generate


class Fair_Dataset(Dataset):
    def __init__(self, X, Y, A, weight=None):
        self.X = X
        self.Y = Y
        self.A = A
        self.weight = weight
        self.data_info = self.get_data_info( Y, A )

    def __getitem__(self, index):
        X = self.X[index]
        Y = self.Y[index]
        A = self.A[index]
        if self.weight is not None:
            assert len(self.weight) == self.X.shape[0]
            weight = self.weight[index]
            return (X, Y, A, weight)
        return (X, Y, A)

    def __len__(self):
        return self.X.shape[0]
    
    def dim(self):
        return self.X.shape[1:]
    
    def get_data_info(self, Y, A ):
        unique_A = np.unique(A)
        unique_Y = np.unique(Y)
        df_YA = pd.DataFrame({'Y': Y.ravel(), 'A': A.ravel()})

        print(Y.shape, A.shape)

        info = pd.DataFrame(index=unique_A, columns=unique_Y)
        for a in unique_A:
            for y in unique_Y:
                info.at[a, y] = df_YA[(df_YA['A'] == a) & (df_YA['Y'] == y)].shape[0]
        return info
    
def mkdir(*args: str) -> tuple:
    for path in args:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
    return args

def get_data_info(train_data, val_data, test_data):
    train_num = sum(list(train_data['num_samples'].values()))
    val_num = sum(list(val_data['num_samples'].values()))
    test_num = sum(list(test_data['num_samples'].values()))
    Ylabel = []
    Alabel = []
    for cdata in train_data['user_data'].values():
        Ylabel.append(np.unique(cdata.Y))
        Alabel.append(np.unique(cdata.A))

    for cdata in val_data['user_data'].values():
        Ylabel.append(np.unique(cdata.Y))
        Alabel.append(np.unique(cdata.A))

    for cdata in test_data['user_data'].values():
        Ylabel.append(np.unique(cdata.Y))
        Alabel.append(np.unique(cdata.A))

    Ylabel = np.unique(np.hstack(Ylabel))
    Alabel = np.unique(np.hstack(Alabel))

    A_info = {i:0 for i in list(Alabel)}
    val_A_info = copy.deepcopy(A_info)
    val_Y1_A_info = copy.deepcopy(A_info)
    val_Y0_A_info = copy.deepcopy(A_info)

    Client_A_info = {client:{i:0 for i in list(Alabel)} for client in train_data['users']}
    Client_A_Y1_info = copy.deepcopy(Client_A_info)

    train_client_A_info = copy.deepcopy(Client_A_info)
    val_client_A_info = copy.deepcopy(Client_A_info)
    test_client_A_info = copy.deepcopy(Client_A_info)
    train_client_Y1_A_info = copy.deepcopy(Client_A_info)
    val_client_Y1_A_info = copy.deepcopy(Client_A_info)
    test_client_Y1_A_info = copy.deepcopy(Client_A_info)
    train_client_Y0_A_info = copy.deepcopy(Client_A_info)
    val_client_Y0_A_info = copy.deepcopy(Client_A_info)
    test_client_Y0_A_info = copy.deepcopy(Client_A_info)


    for c in test_data['users']:
        cdata = test_data['user_data'][c]
        for i in list(Alabel):
            A_info[i] += sum(cdata.A == i)
            test_client_A_info[c][i] += sum(cdata.A == i)
            test_client_Y1_A_info[c][i] = sum((cdata.A == i) * (cdata.Y == 1)) 
            test_client_Y0_A_info[c][i] = sum((cdata.A == i) * (cdata.Y == 0)) 
    
    for c in val_data['users']:
        cdata = val_data['user_data'][c]
        for i in list(Alabel):
            A_info[i] += sum(cdata.A == i) 
            val_A_info[i] += sum(cdata.A == i) 
            val_Y1_A_info[i] = sum((cdata.A == i) * (cdata.Y == 1)) 
            val_Y0_A_info[i] = sum((cdata.A == i) * (cdata.Y == 0)) 
            val_client_A_info[c][i] += sum(cdata.A == i)
            val_client_Y1_A_info[c][i] = sum((cdata.A == i) * (cdata.Y == 1)) 
            val_client_Y0_A_info[c][i] = sum((cdata.A == i) * (cdata.Y == 0)) 
            
    for c in train_data['users']:
        cdata = train_data['user_data'][c]
        for i in list(Alabel):
            A_info[i] += sum(cdata.A == i) 
            train_client_A_info[c][i] += sum(cdata.A == i)
            Client_A_info[c][i] = sum(cdata.A == i) 
            Client_A_Y1_info[c][i] = sum((cdata.A == i) * (cdata.Y == 1)) 
            train_client_Y1_A_info[c][i] = sum((cdata.A == i) * (cdata.Y == 1)) 
            train_client_Y0_A_info[c][i] = sum((cdata.A == i) * (cdata.Y == 0)) 

    A_num = len(Alabel)
    Y_num = len(Ylabel)

    print(A_info[0])

    return {'data_num':train_num + val_num + test_num,
            'train_num':train_num, 'val_num':val_num, 'test_num':test_num, 'Ylabel':Ylabel, 'Alabel':Alabel, 'A_num':A_num, 'Y_num':Y_num, 
            'train_samples': train_data['num_samples'], 'test_samples': test_data['num_samples'],'val_samples':val_data['num_samples'],
            'client_samples':[train_data['num_samples'][i] + test_data['num_samples'][i] + val_data['num_samples'][i] for i in train_data['users']], 
            'A_info':A_info,
            'val_A_info':val_A_info, 'val_Y1_A_info':val_Y1_A_info, 'val_Y0_A_info':val_Y0_A_info,
            'Client_A_info':Client_A_info, 'Client_A_Y1_info':Client_A_Y1_info,
            'train_client_A_info':train_client_A_info,'val_client_A_info':val_client_A_info,'test_client_A_info':test_client_A_info,
            'train_client_Y1_A_info':train_client_Y1_A_info,'val_client_Y1_A_info':val_client_Y1_A_info,'test_client_Y1_A_info':test_client_Y1_A_info,
            'train_client_Y0_A_info':train_client_Y0_A_info,'val_client_Y0_A_info':val_client_Y0_A_info,'test_client_Y0_A_info':test_client_Y0_A_info}

def get_data(options):
    """ 
    Returns train and test datasets:
    """

    data_name = options['data'].lower()
    data_settings = options['data_setting']
    data_settings.update({'num_users':options['num_users']})
    options.update(data_settings)

    if data_name == 'adult':
        if data_settings.get('natural', False):
            train_path = "data/adult/split_data/normal/train.json"
            test_path = "data/adult/split_data/normal/test.json"
            train_data, test_data = read_data(train_path, 'adult', data_settings['sensitive_attr']), read_data(test_path, 'adult', data_settings['sensitive_attr'])
        elif data_settings['dirichlet']:
            train_path = "data/adult/raw_data/train.csv"
            test_path = "data/adult/raw_data/test.csv"
            save_path = f"data/adult/split_data/num_users={data_settings['num_users']} sensitive_attr={data_settings['sensitive_attr']} dirichlet={data_settings['alpha']} by_sensitive={data_settings['by sensitive']}/"
            options['data_save_path'] = save_path
            
            split_data_path = save_path + "split_data.json"

            if os.path.exists(split_data_path) and not data_settings.get('generate',False):
                train_data, val_data, test_data = read_data(split_data_path)
                options['data_exist'] = 1
            else:
                mkdir(save_path)
                adult_process()
                df = pd.concat([pd.read_csv(train_path),pd.read_csv(test_path)], axis=0)
                X, Y = df.drop('salary', axis=1).to_numpy().astype(np.float32),  df['salary'].to_numpy().astype(np.float32)
                colname = df.drop('salary', axis=1).columns.tolist()
                if data_settings['sensitive_attr'] == 'sex-race':
                    X, A, Y = adult_get_sensitive_feature(X, colname, data_settings['sensitive_attr'], Y)
                else:
                    X, A = adult_get_sensitive_feature(X, colname, data_settings['sensitive_attr'])
                if data_settings.get('by sensitive', False):
                    partition, stats = dirichlet(X, A, data_settings['num_users'], data_settings['alpha'])
                    split_data = split(partition['data_indices'], X, Y, A)
                    with open(split_data_path,'w') as outfile:
                        json.dump(split_data, outfile)
                
                    train_data, val_data, test_data = read_data(split_data_path)
                else:
                    partition, stats = dirichlet(X, Y, data_settings['num_users'], data_settings['alpha'])
                    split_data = split(partition['data_indices'], X, Y, A)
                    with open(split_data_path,'w') as outfile:
                        json.dump(split_data, outfile)
                
                    train_data, val_data, test_data = read_data(split_data_path)

    elif data_name == 'celeba':
        if data_settings.get('generate', False) and data_settings.get('natural', False):
            celeba_generate()
        if data_settings['dirichlet']:
            sample_num = 100000
            save_path = f"data/celeba/split_data/sample_num={sample_num} num_users={data_settings['num_users']} sensitive_attr={data_settings['sensitive_attr']} dirichlet={data_settings['alpha']} by_sensitive={data_settings['by sensitive']}/"
            options['data_save_path'] = save_path
            split_train_path = save_path + "train.npy"
            split_test_path = save_path + "test.npy"
            split_data_path = save_path + "split_data.npy"

            if os.path.exists(split_data_path) and not data_settings.get('generate',False):
                train_data, val_data, test_data = read_data(split_data_path, name='celeba')
                options['data_exist'] = 1
                print('celeba data processed.')
            else:
                mkdir(save_path)
                if data_settings['sensitive_attr'] == 'sex':
                    sensitive_attr = 'Male'
                elif data_settings['sensitive_attr'] == 'age':
                    sensitive_attr = 'Young'
                elif data_settings['sensitive_attr'] == 'sex-race':
                    sensitive_attr = ['Male', 'Pale_Skin']
                elif data_settings['sensitive_attr'] == 'race':
                    sensitive_attr = 'Pale_Skin'
                X, Y, A = celeba_data_processing(sensitive_attr, sample_num)

                if data_settings.get('by sensitive', False):
                    partition, stats = dirichlet(X, A, data_settings['num_users'], data_settings['alpha'])
                    split_data = celeba_split(partition['data_indices'], X, Y, A)
                    # del X, Y, A
                    # np.save(split_data_path, split_data)

                else:
                    partition, stats = dirichlet(X, Y, data_settings['num_users'], data_settings['alpha'])
                    split_data = celeba_split(partition['data_indices'], X, Y, A)
                    # del X, Y, A
                    # np.save(split_data_path, split_data)

                train_data, val_data, test_data = celeba_read_data(split_data, name='celeba')
                print('celeba data processed.')
        

    elif data_name == 'compas':
        save_path = f"data/compas/split_data/num_users={data_settings['num_users']} sensitive_attr={data_settings['sensitive_attr']} dirichlet={data_settings['alpha']} by_sensitive={data_settings['by sensitive']}/"
        options['data_save_path'] = save_path
        split_train_path = save_path + "train.json"
        split_test_path = save_path + "test.json"
        split_data_path = save_path + "split_data.json"
        print("use compas.")

        if os.path.exists(split_data_path) and not data_settings.get('generate',False):
            train_data, val_data, test_data = read_data(split_data_path)
            options['data_exist'] = 1
        else:
            mkdir(save_path)
            X, Y, A = compas_1_data_processing(data_settings['sensitive_attr'])
            
            if data_settings.get('by sensitive', False):
                partition, stats = dirichlet(X, A, data_settings['num_users'], data_settings['alpha'])
                split_data = split(partition['data_indices'], X, Y, A)

                with open(split_data_path,'w') as outfile:
                        json.dump(split_data, outfile)
            
                train_data, val_data, test_data = read_data(split_data_path)

            else:
                partition, stats = dirichlet(X, Y, data_settings['num_users'], data_settings['alpha'])
                split_data = split(partition['data_indices'], X, Y, A)

                with open(split_data_path,'w') as outfile:
                        json.dump(split_data, outfile)
            
                train_data, val_data, test_data = read_data(split_data_path)
    
    elif data_name == 'enem':
        save_path = f"data/enem/split_data/num_users={data_settings['num_users']} sensitive_attr={data_settings['sensitive_attr']} dirichlet={data_settings['alpha']} by_sensitive={data_settings['by sensitive']}/"
        options['data_save_path'] = save_path
        split_train_path = save_path + "train.json"
        split_test_path = save_path + "test.json"
        split_data_path = save_path + "split_data.json"
        print("use enem.")

        if os.path.exists(split_data_path) and not data_settings.get('generate',False):
            train_data, val_data, test_data = read_data(split_data_path)
            options['data_exist'] = 1
        else:
            mkdir(save_path)
            X, Y, A = enem_process(data_settings['sensitive_attr'])
            
            if data_settings.get('by sensitive', False):
                partition, stats = dirichlet(X, A, data_settings['num_users'], data_settings['alpha'])
                split_data = split(partition['data_indices'], X, Y, A)

                with open(split_data_path,'w') as outfile:
                        json.dump(split_data, outfile)
            
                train_data, val_data, test_data = read_data(split_data_path)

            else:
                partition, stats = dirichlet(X, Y, data_settings['num_users'], data_settings['alpha'])
                split_data = split(partition['data_indices'], X, Y, A)

                with open(split_data_path,'w') as outfile:
                        json.dump(split_data, outfile)
            
                train_data, val_data, test_data = read_data(split_data_path)

    elif data_name == 'synth':
        save_path = f"data/synth/split_data/num_users={data_settings['num_users']} sensitive_attr={data_settings['sensitive_attr']} dirichlet={data_settings['alpha']} by_sensitive={data_settings['by sensitive']}/"
        options['data_save_path'] = save_path
        split_train_path = save_path + "train.json"
        split_test_path = save_path + "test.json"
        split_data_path = save_path + "split_data.json"
        print("use synth.")

        mkdir(save_path)
        from .synth_data import synth_process
        data_num = options['data_setting'].get('data_num', 5000)
        client_data = synth_process(data_num)

        split_train = {'users': [], 'user_data':{}, 'num_samples':{}}
        split_val = copy.deepcopy(split_train)
        split_test = copy.deepcopy(split_train)

        for client in ['0','1']:
            split_train['users'].append(client)
            split_val['users'].append(client)
            split_test['users'].append(client)
            
            X_train = np.array(client_data['client_'+client]["X_train_removed"]).astype(np.float32)
            Y_train = np.array(client_data['client_'+client]["y_train"]).astype(np.float32).reshape(-1,1)
            A_train = np.array(client_data['client_'+client]["X_train"][:,2]).astype(np.float32).reshape(-1,1)

            X_test = np.array(client_data['client_'+client]["X_test_removed"]).astype(np.float32)
            Y_test = np.array(client_data['client_'+client]["y_test"]).astype(np.float32).reshape(-1,1)
            A_test = np.array(client_data['client_'+client]["X_test"][:,2]).astype(np.float32).reshape(-1,1)
            
            split_train['user_data'][client] = Fair_Dataset(X_train, Y_train, A_train)
            split_val['user_data'][client] = Fair_Dataset(X_train, Y_train, A_train)
            split_test['user_data'][client] = Fair_Dataset(X_test, Y_test, A_test)

            split_train['num_samples'][client] = X_train.shape[0]
            split_val['num_samples'][client] = X_train.shape[0]
            split_test['num_samples'][client] = X_test.shape[0]
        
            train_data, val_data, test_data = split_train,split_val,split_test
    
    else:
        raise ValueError('Not support dataset {}!'.format(data_name))
    
    options['num_shape'] = train_data['user_data'][list(train_data['user_data'])[0]].dim()
    data_info = get_data_info(train_data, val_data, test_data)
    options['data_info'] = data_info
    print(data_info)
    return train_data, val_data, test_data



def enem_process(sensitive_attr):
    enem_path = 'data/enem/raw_data/microdados_enem_2020/DADOS/' #changed to 2020
    enem_file = 'MICRODADOS_ENEM_2020.csv' #changed for 2020
    label = ['NU_NOTA_CH'] ## Labels could be: NU_NOTA_CH=human science, NU_NOTA_LC=languages&codes, NU_NOTA_MT=math, NU_NOTA_CN=natural science
    group_attribute = ['TP_COR_RACA','TP_SEXO']
    question_vars = ['Q00'+str(x) if x<10 else 'Q0' + str(x) for x in range(1,25)] #changed for 2020
    domestic_vars = ['SG_UF_PROVA', 'TP_FAIXA_ETARIA'] #changed for 2020
    all_vars = label+group_attribute+question_vars+domestic_vars

    n_sample = 1400000
    n_classes = 2

    fname = 'data/enem/processed_data/enem-'+str(n_sample)+'-20.pkl'
    if os.path.isfile(fname):
        df = pd.read_pickle(fname)
    else:
        # df = load_enem(enem_path, enem_file, all_vars, label, n_sample)
        df = load_enem(enem_path, enem_file, all_vars, label, n_sample, n_classes, multigroup=False)
        df.to_pickle(fname)

    df['gradebin'] = df['gradebin'].astype(int)

    # start_time = time.localtime()
    # start_time_str = strftime("%Y-%m-%d-%H.%M.%S", start_time)
    # filename = 'enem-'+ str(df.shape[0]) +'-mp-' + start_time_str
    # f = open(filename+'-log.txt','w')

    # repetition = 10
    # use_protected = True
    # use_sample_weight = True
    # tune_threshold = False
    # # tolerance = [0.000, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    # tolerance = [0.000, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]

    label_name = 'gradebin'
    protected_attrs = 'racebin' if sensitive_attr=='race' else 0

    X = df.drop(columns=[label_name,protected_attrs]).to_numpy()
    A = df[protected_attrs].to_numpy()
    Y = df[label_name].to_numpy()

    return X,A,Y


def get_idx_wo_protected(feature_names, protected_attrs):
    idx_wo_protected = set(range(len(feature_names)))
    protected_attr_idx = [feature_names.index(x) for x in protected_attrs]
    idx_wo_protected = list(idx_wo_protected - set(protected_attr_idx))
    return idx_wo_protected

def get_idx_w_protected(feature_names):
    return list(set(range(len(feature_names))))

def get_idx_protected(feature_names, protected_attrs):
    protected_attr_idx = [feature_names.index(x) for x in protected_attrs]
    idx_protected = list(set(protected_attr_idx))
    return idx_protected

def load_enem(file_path, filename, features, grade_attribute, n_sample, n_classes, multigroup=False):
    ## load csv
    df = pd.read_csv(file_path+filename, encoding='cp860', sep=';')
    # print('Original Dataset Shape:', df.shape)

    ## Remove all entries that were absent or were eliminated in at least one exam
    ix = ~df[['TP_PRESENCA_CN', 'TP_PRESENCA_CH', 'TP_PRESENCA_LC', 'TP_PRESENCA_MT']].applymap(lambda x: False if x == 1.0 else True).any(axis=1)
    df = df.loc[ix, :]

    ## Remove "treineiros" -- these are individuals that marked that they are taking the exam "only to test their knowledge". It is not uncommon for students to take the ENEM in the middle of high school as a dry run
    df = df.loc[df['IN_TREINEIRO'] == 0, :]

    ## drop eliminated features
    df.drop(['TP_PRESENCA_CN', 'TP_PRESENCA_CH', 'TP_PRESENCA_LC', 'TP_PRESENCA_MT', 'IN_TREINEIRO'], axis=1, inplace=True)

    ## subsitute race by names
    # race_names = ['N/A', 'Branca', 'Preta', 'Parda', 'Amarela', 'Indigena']
    race_names = [np.nan, 'Branca', 'Preta', 'Parda', 'Amarela', 'Indigena']
    df['TP_COR_RACA'] = df.loc[:, ['TP_COR_RACA']].applymap(lambda x: race_names[x]).copy()

    ## remove repeated exam takers
    ## This pre-processing step significantly reduces the dataset.
    df = df.loc[df.TP_ST_CONCLUSAO.isin([1])]

    ## select features
    df = df[features]

    ## Dropping all rows or columns with missing values
    df = df.dropna()

    ## Creating racebin & gradebin & sexbin variable
    df['gradebin'] = construct_grade(df, grade_attribute, n_classes)
    if multigroup:
        df['racebin'] = construct_race(df, 'TP_COR_RACA')
    else:
        df['racebin'] =np.logical_or((df['TP_COR_RACA'] == 'Branca').values, (df['TP_COR_RACA'] == 'Amarela').values).astype(int)
    df['sexbin'] = (df['TP_SEXO'] == 'M').astype(int)

    df.drop([grade_attribute[0], 'TP_COR_RACA', 'TP_SEXO'], axis=1, inplace=True)

    ## encode answers to questionaires
    ## Q005 is 'Including yourself, how many people currently live in your household?'
    question_vars = ['Q00' + str(x) if x < 10 else 'Q0' + str(x) for x in range(1, 25)]
    for q in question_vars:
        if q != 'Q005':
            df_q = pd.get_dummies(df[q], prefix=q)
            df.drop([q], axis=1, inplace=True)
            df = pd.concat([df, df_q.iloc[:, :-1]], axis=1)
            
    ## check if age range ('TP_FAIXA_ETARIA') is within attributes
    if 'TP_FAIXA_ETARIA' in features:
        q = 'TP_FAIXA_ETARIA'
        df_q = pd.get_dummies(df[q], prefix=q)
        df.drop([q], axis=1, inplace=True)
        df = pd.concat([df, df_q.iloc[:, :-1]], axis=1)

    ## encode SG_UF_PROVA (state where exam was taken)
    df_res = pd.get_dummies(df['SG_UF_PROVA'], prefix='SG_UF_PROVA')
    df.drop(['SG_UF_PROVA'], axis=1, inplace=True)
    df = pd.concat([df, df_res], axis=1)

    df = df.dropna()
    ## Scaling ##
    scaler = MinMaxScaler()
    scale_columns = list(set(df.columns.values) - set(['gradebin', 'racebin']))
    df[scale_columns] = pd.DataFrame(scaler.fit_transform(df[scale_columns]), columns=scale_columns, index=df.index)
    # print('Preprocessed Dataset Shape:', df.shape)

    df = df.sample(n=min(n_sample, df.shape[0]), axis=0, replace=False)
    return df

def construct_race(df, protected_attribute):
    race_dict = {'Branca': 1, 'Preta': 2, 'Parda': 3, 'Amarela': 4, 'Indigena': 5} # changed to match ENEM 2020 numbering
    return df[protected_attribute].map(race_dict)

def construct_grade(df, grade_attribute, n):
    v = df[grade_attribute[0]].values
    quantiles = np.nanquantile(v, np.linspace(0.0, 1.0, n+1))
    return pd.cut(v, quantiles, labels=np.arange(n))



def adult_process():
    # Adult
    sensitive_attributes = ['sex']
    categorical_attributes = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'native-country']
    continuous_attributes = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
    features_to_keep = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss','hours-per-week', 
                'native-country', 'salary']
    label_name = 'salary'

    adult = process_adult_csv('D:/code/fed fairness/FedFairPost_2/data/adult/raw_data/adult.data', label_name, ' >50K', sensitive_attributes, [' Female'], categorical_attributes, continuous_attributes, features_to_keep, na_values = [], header = None, columns = features_to_keep)
    test = process_adult_csv('D:/code/fed fairness/FedFairPost_2/data/adult/raw_data/adult.test', label_name, ' >50K.', sensitive_attributes, [' Female'], categorical_attributes, continuous_attributes, features_to_keep, na_values = [], header = None, columns = features_to_keep) # the distribution is very different from training distribution
    test['native-country_ Holand-Netherlands'] = 0
    test = test[adult.columns]

    adult_num_features = len(adult.columns)-1

    adult.to_csv('data/adult/raw_data/train.csv', index=None)
    test.to_csv('data/adult/raw_data/test.csv', index=None)
    
def process_adult_csv(filename, label_name, favorable_class, sensitive_attributes, privileged_classes, categorical_attributes, continuous_attributes, features_to_keep, na_values = [], header = 'infer', columns = None):
    """
    from https://github.com/yzeng58/Improving-Fairness-via-Federated-Learning/blob/main/FedFB/DP_load_dataset.py
    process the adult file: scale, one-hot encode
    only support binary sensitive attributes -> [gender, race] -> 4 sensitive groups 
    """
    skiprows = 1 if filename.endswith('test') else 0
    df = pd.read_csv(os.path.join(filename), delimiter = ',', header = header, na_values = na_values, skiprows=skiprows)
    if header == None: df.columns = columns
    df = df[features_to_keep]

    # apply one-hot encoding to convert the categorical attributes into vectors
    df = pd.get_dummies(df, columns = categorical_attributes)

    # normalize numerical attributes to the range within [0, 1]
    def scale(vec):
        minimum = min(vec)
        maximum = max(vec)
        return (vec-minimum)/(maximum-minimum)
    
    df[continuous_attributes] = df[continuous_attributes].apply(scale, axis = 0)
    df.loc[df[label_name] != favorable_class, label_name] = 0
    df.loc[df[label_name] == favorable_class, label_name] = 1
    df[label_name] = df[label_name].astype('category').cat.codes
    df['sex'] = df['sex'].map({' Male':0, ' Female':1}).astype('category')
    return df


def compas_1_data_processing(sensitive='sex-race'):
    #@title Load COMPAS dataset

    LABEL_COLUMN = 'two_year_recid'
    if sensitive == 'sex-race':
        sensitive_attributes = ['sex_Female', 'race_African-American']
    elif sensitive == 'race':
        sensitive_attributes = ['race_African-American']


    def get_data():
        data_path = "data/compas/raw_data/compas-scores-two-years.csv"
        df = pd.read_csv(data_path)
        FEATURES = [
            'age', 'c_charge_degree', 'race', 'age_cat', 'score_text', 'sex',
            'priors_count', 'days_b_screening_arrest', 'decile_score', 'is_recid',
            'two_year_recid'
        ]
        df = df[FEATURES]
        df = df[df.days_b_screening_arrest <= 30]
        df = df[df.days_b_screening_arrest >= -30]
        df = df[df.is_recid != -1]
        df = df[df.c_charge_degree != 'O']
        df = df[df.score_text != 'N/A']
        continuous_features = [
            'priors_count', 'days_b_screening_arrest', 'is_recid', 'two_year_recid'
        ]
        continuous_to_categorical_features = ['age', 'decile_score', 'priors_count']
        categorical_features = ['c_charge_degree', 'race', 'score_text', 'sex']
        # continuous_to_categorical_features = [ 'priors_count']
        # categorical_features = ['c_charge_degree', 'race', 'sex']

        # Functions for preprocessing categorical and continuous columns.
        def binarize_categorical_columns(input_df, categorical_columns=[]):
            # Binarize categorical columns.
            binarized_df = pd.get_dummies(input_df, columns=categorical_columns)
            return binarized_df

        def bucketize_continuous_column(input_df, continuous_column_name, bins=None):
            input_df[continuous_column_name] = pd.cut(
                input_df[continuous_column_name], bins, labels=False)

        for c in continuous_to_categorical_features:
            b = [0] + list(np.percentile(df[c], [20, 40, 60, 80, 90, 100]))
            if c == 'priors_count':
                b = list(np.percentile(df[c], [0, 50, 70, 80, 90, 100]))
            bucketize_continuous_column(df, c, bins=b)

        # df = binarize_categorical_columns(
        #     df,
        #     categorical_columns=categorical_features)

        df = binarize_categorical_columns(
            df,
            categorical_columns=categorical_features +
            continuous_to_categorical_features)

        to_fill = [
            u'decile_score_0', u'decile_score_1', u'decile_score_2',
            u'decile_score_3', u'decile_score_4', u'decile_score_5'
        ]
        for i in range(len(to_fill) - 1):
            df[to_fill[i]] = df[to_fill[i:]].max(axis=1)
            
        to_fill = [
            u'priors_count_0.0', u'priors_count_1.0', u'priors_count_2.0',
            u'priors_count_3.0', u'priors_count_4.0'
        ]
        for i in range(len(to_fill) - 1):
            df[to_fill[i]] = df[to_fill[i:]].max(axis=1)

        print(df.columns)
        features = [
            u'days_b_screening_arrest', u'c_charge_degree_F', u'c_charge_degree_M',
            u'race_African-American', u'race_Asian', u'race_Caucasian',
            u'race_Hispanic', u'race_Native American', u'race_Other',
            u'score_text_High', u'score_text_Low', u'score_text_Medium',
            u'sex_Female', u'sex_Male', u'age_0', u'age_1', u'age_2', u'age_3',
            u'age_4', u'age_5', u'decile_score_0', u'decile_score_1',
            u'decile_score_2', u'decile_score_3', u'decile_score_4',
            u'decile_score_5', u'priors_count_0.0', u'priors_count_1.0',
            u'priors_count_2.0', u'priors_count_3.0', u'priors_count_4.0'
        ]

        # # new
        # features = [
        #     u'days_b_screening_arrest', u'c_charge_degree_F', u'c_charge_degree_M',
        #     u'race_African-American', u'race_Asian', u'race_Caucasian',
        #     u'race_Hispanic', u'race_Native American', u'race_Other',
        #     u'sex_Female', u'sex_Male', u'age', u'priors_count_0.0', u'priors_count_1.0',
        #     u'priors_count_2.0', u'priors_count_3.0', u'priors_count_4.0'
        # ]
        # print(len(features))

        label = ['two_year_recid']

        df = df[features + label]
        return df, features, label

    df, feature_names, label_column = get_data()

    # if sensitive == 'race':
    #     df_w = df[df['race_Caucasian'] == 1]
    #     df_b = df[df['race_African-American'] == 1]
    #     df = pd.concat([df_w, df_b])

    from sklearn.utils import shuffle
    df = shuffle(df)
    N = len(df)
    # train_df = df[:int(N * 0.66)]
    # test_df = df[int(N * 0.66):]

    X_compas = np.array(df[feature_names])
    y_compas = np.array(df[label_column]).flatten()
    # X_test_compas = np.array(test_df[feature_names])
    # y_test_compas = np.array(test_df[label_column]).flatten()

    if sensitive == 'sex-race':

        # 0: male non-black, 1: female non-black, 2: male black, 3: female black
        A_compas = np.array(df[sensitive_attributes[0]] + df[sensitive_attributes[1]] * 2).flatten()
        # A_test_compas = np.array(test_df[sensitive_attributes[0]] + test_df[sensitive_attributes[1]] * 2).flatten()

        sex_race_idx = [i for i, value in enumerate(feature_names) if (value.startswith('race') or value.startswith('sex')) ==True]
        X_compas = np.delete(X_compas, sex_race_idx, axis=1)
        # X_test_compas = np.delete(X_test_compas, sex_race_idx, axis=1)

        print(X_compas.shape)
    
    elif sensitive == 'race':
        # 0: non-black, 1: black
        A_compas = np.array(df[sensitive_attributes]).flatten()
        # A_test_compas = np.array(test_df[sensitive_attributes]).flatten()

        sen_idx = [i for i, value in enumerate(feature_names) if value.startswith('race')==True]
        X_compas = np.delete(X_compas, sen_idx, axis=1)
        # X_test_compas = np.delete(X_test_compas, sen_idx, axis=1)

    print("compas process end.")

    return X_compas, y_compas,  A_compas

def celeba_data_processing(sensitive_attr, sample_num, batch_size=32, mmap_file="processed_data.mmap"):

    path = os.path.join('data','celeba', 'processed_data')
    file_name = os.path.join(path,f'num={sample_num}_celeba.npy')
    if os.path.exists(file_name):
        loaded_data = np.load(file_name, allow_pickle=True).item()
        X = loaded_data['X']
        Y = loaded_data['Y']
        A = loaded_data['A']
    else:
        if not os.path.exists(path):
            mkdir(path)
        f_identities = open(os.path.join('data', 'celeba', 'raw_data', 'identity_CelebA.txt'), 'r')
        identities = f_identities.read().split('\n')

        f_attributes = open(os.path.join('data', 'celeba', 'raw_data', 'list_attr_celeba.txt'), 'r')
        attributes = f_attributes.read().split('\n')

        tar = 'Smiling'
        sen_attr = sensitive_attr

        target_idx = attributes[1].split().index(tar)
        if isinstance(sen_attr, list):
            assert len(sen_attr) == 2
            sen_idx = [attributes[1].split().index(sen) for sen in sen_attr]
        elif isinstance(sen_attr, str):
            sen_idx = attributes[1].split().index(sen_attr)

        image = {}

        for line in attributes[2:]:
            info = line.split()
            if not info:
                continue
            image_id = info[0]
            tar_img = (int(info[1:][target_idx]) + 1) / 2
            if isinstance(sen_attr, list):
                sen_img1 = (int(info[1:][sen_idx[0]]) + 1) / 2
                sen_img2 = (int(info[1:][sen_idx[1]]) + 1) / 2
                sen_img = sen_img1 + 2 * sen_img2
            elif isinstance(sen_attr, str):
                sen_img = (int(info[1:][sen_idx]) + 1) / 2

            image[image_id] = tar_img, sen_img

        images_path = Path(os.path.join('data', 'celeba', 'raw_data', 'img_align_celeba'))
        images_list = list(images_path.glob('*.jpg'))
        images_list_str = [str(x) for x in images_list]
        images_ids = random.sample(images_list_str, sample_num)

        sample_target = []
        sample_sensitive = []
        for path in images_ids:
            sample_target.append(image[path[-10:]][0])
            sample_sensitive.append(image[path[-10:]][1])

        transform = transforms.Compose([
            transforms.CenterCrop((178, 178)), 
            transforms.Resize((128, 128)), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
        ])

        print('start.')

        shape = (sample_num, 3, 128, 128)
        X = np.memmap(mmap_file, dtype=np.float16, mode='w+', shape=shape)

        for i in range(0, len(images_ids), batch_size):
            batch_ids = images_ids[i:i+batch_size]
            sample_target_batch = [image[path[-10:]][0] for path in batch_ids]
            sample_sensitive_batch = [image[path[-10:]][1] for path in batch_ids]

            mp_img_loader = multiprocess_img_load(batch_ids, transform)
            batch_X = mp_img_loader.get_imgs().astype(np.float32)

            X[i:i+len(batch_X)] = batch_X
            sample_target[i:i+len(batch_X)] = sample_target_batch
            sample_sensitive[i:i+len(batch_X)] = sample_sensitive_batch

        print(X.shape)
        print(type(X))
        print(np.max(X),np.min(X))

        print('end.')
        Y, A = np.array(sample_target,dtype=np.float16), np.array(sample_sensitive, dtype=np.float16)

        data_dict = {'X': X, 'Y': Y, 'A': A}
        np.save(file_name, data_dict)
        X.flush()
        os.remove(mmap_file)  # delete memmap 
    return X, Y, A


class multiprocess_img_load(object):
    def __init__(self, img_paths:list, transform, img_size=(3,128,128), n_thread=None) -> None:
        self.image_paths = img_paths
        self.img_size = img_size
        self.num_img = len(img_paths)
        self._mutex_put = threading.Lock()
        self.n_thread = n_thread if (n_thread is not None) else max(1, multiprocessing.cpu_count() - 2)
        self.transform = transform
    
    def get_imgs(self):
        self._buffer = np.zeros([self.num_img]+list(self.img_size))
        batch_size = round(self.num_img / self.n_thread)
        batch_idx = []
        for i in range(self.n_thread):
            idx = list(range(i * batch_size, (i+1) * batch_size)) if (i+1) * batch_size <= self.num_img else list(range(i * batch_size, self.num_img))
            batch_idx.append(idx)
        t_list = []
        for tid in range(self.n_thread):
            img_ids = list(range(tid * batch_size, (tid+1) * batch_size)) if (tid+1) * batch_size <= self.num_img else range(tid * batch_size, self.num_img)
            img_target = [self.image_paths[i] for i in img_ids]
            t = threading.Thread(target=self.load_image, args=(img_target, img_ids))
            t_list.append(t)
            t.start()

        for t in t_list:
            t.join()

        del t_list

        return self._buffer

    def load_image(self, img_names, img_ids):
        batch_images = np.vstack([np.expand_dims(self.transform(Image.open(img)).numpy(), axis=0) for img in img_names])
        self._mutex_put.acquire()
        self._buffer[img_ids] = batch_images
        self._mutex_put.release()

# def celeba_data_processing(sensitive_attr, sample_num):
#     f_identities = open(os.path.join( 'data', 'celeba', 'raw_data', 'identity_CelebA.txt'), 'r')
#     identities = f_identities.read().split('\n')

#     f_attributes = open(os.path.join('data', 'celeba', 'raw_data', 'list_attr_celeba.txt'), 'r')
#     attributes = f_attributes.read().split('\n')

#     tar = 'Smiling'

#     sen_attr = sensitive_attr

#     target_idx = attributes[1].split().index(tar)
#     if type(sen_attr) == list:
#         assert len(sen_attr) == 2
#         sen_idx = [attributes[1].split().index(sen) for sen in sen_attr]
#     elif type(sen_attr) == str:
#         sen_idx = attributes[1].split().index(sen_attr)

#     image = {}

#     for line in attributes[2:]:
#         info = line.split()
#         if len(info) == 0:
#             continue
#         image_id = info[0]
#         tar_img = (int(info[1:][target_idx]) + 1) / 2
#         if type(sen_attr) == list:
#             # 0: non-white female, 1: non-white male, 2: white female, 3:white male
#             sen_img1 = (int(info[1:][sen_idx[0]]) + 1) / 2
#             sen_img2 = (int(info[1:][sen_idx[1]]) + 1) / 2
#             sen_img = sen_img1 + 2 * sen_img2
#         elif type(sen_attr) == str:
#             sen_img = (int(info[1:][sen_idx]) + 1) / 2

#         image[image_id] = tar_img, sen_img

#     images_path = Path(os.path.join('data', 'celeba', 'raw_data', 'img_align_celeba'))

#     images_list = list(images_path.glob('*.jpg')) # list(images_path.glob('*.png'))
#     images_list_str = [ str(x) for x in images_list ]
#     images_ids = random.sample(images_list_str, sample_num)

#     sample_target = []
#     sample_sensitive = []
#     for path in images_ids:
#         sample_target.append(image[path[-10:]][0])
#         sample_sensitive.append(image[path[-10:]][1])

#     transform = transforms.Compose([
#             transforms.CenterCrop((178, 178)), 
#             transforms.Resize((128, 128)), 
#             transforms.ToTensor(),
#             # transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]) 
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
#         ])
    
#     print('start.')

#     mp_img_loader = multiprocess_img_load(images_ids, transform)
#     X = mp_img_loader.get_imgs().astype(np.float32)
#     print(X.shape)
#     print(type(X))

#     # X1 = np.vstack(np.expand_dims([transform(Image.open(img)).numpy() for img in images_ids],axis=0))
#     # print(np.sum(X != X1))
#     print('end.')
#     Y, A = np.array(sample_target), np.array(sample_sensitive)
#     return X,Y,A

# class multiprocess_img_load(object):
#     def __init__(self, img_paths:list,transform, img_size=(3,128,128), n_thread=None) -> None:
#         self.image_paths = img_paths
#         self.img_size = img_size
#         self.num_img = len(img_paths)
#         self._mutex_put = threading.Lock()
#         self.n_thread = n_thread if (n_thread is not None) else max(1, multiprocessing.cpu_count() - 2)
#         self.transform = transform
    
#     def get_imgs(self):
#         self._buffer = np.zeros([self.num_img]+list(self.img_size))
#         batch_size = round(self.num_img / self.n_thread)
#         batch_idx = []
#         for i in range(self.n_thread):
#             idx = list(range(i * batch_size, (i+1) * batch_size)) if (i+1) * batch_size <= self.num_img else list(range(i * batch_size, self.num_img))
#             batch_idx.append(idx)
#         t_list = []
#         for tid in range(self.n_thread):
#             img_ids = list(range(tid * batch_size, (tid+1) * batch_size)) if (tid+1) * batch_size <= self.num_img else range(tid * batch_size, self.num_img)
#             img_target = [self.image_paths[i] for i in img_ids]
#             t = threading.Thread(target=self.load_image, args=(img_target, img_ids))
#             t_list.append(t)
#             t.start()

#         for t in t_list:
#             t.join()

#         del t_list

#         return self._buffer

#     def load_image(self, img_names, img_ids):
#         batch_images = np.vstack([np.expand_dims(self.transform(Image.open(img)).numpy(), axis=0) for img in img_names])
#         self._mutex_put.acquire()
#         self._buffer[img_ids] = batch_images
#         self._mutex_put.release()


def celeba_split(data_indices, X ,Y ,A):
    split_data = {'users': [], 'user_data':{}, 'num_samples':[]}
    for i in range(len(data_indices)):
        split_data['users'].append(str(i))
        split_data['user_data'][str(i)] = {'x':X[data_indices[i],:],
                                      'y':Y[data_indices[i]],
                                      'A':A[data_indices[i]]}
        split_data['num_samples'].append(len(data_indices[i]))
    return split_data

def get_unsaved_data(data_split):
    for client in data_split['user_data']:
        X = np.array(data_split['user_data'][client]["x"]).astype(np.float32)
        Y = np.array(data_split['user_data'][client]["y"]).astype(np.float32).reshape(-1,1)
        A = np.array(data_split['user_data'][client]["A"]).astype(np.float32).reshape(-1,1)
        dataset = Fair_Dataset(X, Y, A)
        data_split['user_data'][client] = dataset
    return data_split

def bank_get_sensitive_feature(X, colname, sensitive_attr):
    if sensitive_attr == 'age':
        attr_idx = colname.index(sensitive_attr)
        A = X[:,attr_idx]
        X = np.delete(X, attr_idx, axis = 1)
    return X,A

def compas_get_sensitive_feature(X, colname, sensitive_attr):
    sex_attr = []
    race_attr = []
    for col in colname:
        if col.startswith('race'):
            race_attr.append(col)
        elif col.startswith('sex'):
            sex_attr.append(col)
    
    if sensitive_attr == 'sex':
        attr_idx = [colname.index(attr) for attr in sex_attr]
        A = np.argmax(X[:,attr_idx], axis =1 )  # [1: Male, 0: Female]
        X = np.delete(X, attr_idx, axis = 1)
    elif sensitive_attr == 'race':
        attr_idx = [colname.index(attr) for attr in race_attr]
        A = np.argmax(X[:,attr_idx], axis = 1) # ['African-American': 0,'Caucasian': 1,'Asian':2,'Hispanic':3]
        A[A>=1] = 1
        X = np.delete(X, attr_idx, axis = 1)
    elif sensitive_attr == 'non-sex':
        attr_idx = [colname.index(attr) for attr in sex_attr]
        A = np.argmax(X[:,attr_idx], axi = 1) 
    elif sensitive_attr == 'non-race':
        attr_idx = [colname.index(attr) for attr in race_attr] 
        A = np.argmax(X[:,attr_idx], axis = 1)
    return X, A

def split_celeba_data(ids: list):
    path = 'data/celeba/raw_data/img_align_celeba/'
    imgs = np.concatenate([np.expand_dims(np.array(Image.open(path + id)).transpose(2,0,1), axis=0) for id in ids], axis=0)
    
    return imgs

def partition_test_data(separation, targets):
    label_num = len(set(targets))
    targets_numpy = np.array(targets, dtype=np.int32)
    data_indices = [[] for _ in range(len(separation[0]))]
    data_idx_for_each_label = [
        np.where(targets_numpy == i)[0] for i in range(label_num)
    ]
    for k in range(label_num):
        distrib_cumsum = (np.cumsum(separation[k]) * len(data_idx_for_each_label[k])).astype(int)[:-1]
        data_indices = [
            np.concatenate((idx_j, idx.tolist())).astype(np.int64)
            for idx_j, idx in zip(
                data_indices, np.split(data_idx_for_each_label[k], distrib_cumsum)
            )
        ]
    
    return data_indices

def split(data_indices, X ,Y ,A):
    split_data = {'users': [], 'user_data':{}, 'num_samples':{}}
    for i in range(len(data_indices)):
        split_data['users'].append(str(i))
        split_data['user_data'][str(i)] = {'x':X[data_indices[i],:].tolist(),
                                      'y':Y[data_indices[i]].tolist(),
                                      'A':A[data_indices[i]].tolist()}
        split_data['num_samples'][str(i)] = len(data_indices[i])
    return split_data


def adult_get_sensitive_feature(X, colname, sensitive, Y=None):
    sex_attr = 'sex'
    race_attr = []
    for col in colname:
        if col.startswith('race'):
            race_attr.append(col)
    if sensitive == "race":
        attr = 'race_ White'
        attr_idx = colname.index(attr)
        A = np.array(X[:,attr_idx])  
        # print(np.unique(A))
        del_idx = [colname.index(attr) for attr in race_attr]
        X = np.delete(X, del_idx, axis = 1)
    elif sensitive == "sex":
        attr_idx = colname.index(sex_attr)
        A = X[:, attr_idx] # [1: female, 0: male]
        X = np.delete(X, attr_idx, axis = 1)
    elif sensitive == "none-race":
        attr_idx = [colname.index(attr) for attr in race_attr]
        A = np.argmax(X[:,attr_idx], axis =1 ) 
    elif sensitive == "none-sex":
        attr_idx = colname.index(sex_attr)
        A = X[:, attr_idx] # [1: female, 0: male]
    elif sensitive == "sex-race":
        race_idx = [colname.index(attr) for attr in race_attr] 
        race_unused = [colname.index(attr) for attr in ['race_ Amer-Indian-Eskimo', 'race_ Asian-Pac-Islander', 'race_ Other']] 
        Y = Y[np.sum(X[:,race_unused],axis=1) == 0]
        X = X[np.sum(X[:,race_unused],axis=1) == 0,:]
        sex_idx = colname.index(sex_attr)
        A = (np.argmax(X[:,race_idx], axis =1) + X[:,sex_idx]) - 2
        X = np.delete(X, race_idx + [sex_idx], axis = 1)
        return X,A,Y


    else:
        print("error sensitive attr")
        exit()
    
    return X, A

def read_data(path, name=None, sensitive_process=None):
    split_train = {'users': [], 'user_data':{}, 'num_samples':{}}
    split_val = copy.deepcopy(split_train)
    split_test = copy.deepcopy(split_train)

    if name == 'celeba':
        data_split = np.load(path, allow_pickle=True).item()
    else:
        with open(path, 'rb') as file:
            data_split = json.load(file)

    for client in data_split['users']:
        split_train['users'].append(client)
        split_val['users'].append(client)
        split_test['users'].append(client)

        X = np.array(data_split['user_data'][client]["x"]).astype(np.float32)

        Y = np.array(data_split['user_data'][client]["y"]).astype(np.float32).reshape(-1,1)

        A = np.array(data_split['user_data'][client]["A"]).astype(np.float32).reshape(-1,1)

        n = np.arange(X.shape[0])
        indices = np.random.permutation(n)
        train_index, val_index, test_index = indices[:int(len(n)*0.7)], indices[int(len(n)*0):int(len(n)*0.7)], indices[int(len(n)*0.7):]
        split_train['user_data'][client] = Fair_Dataset(X[train_index,:], Y[train_index,:], A[train_index,:])
        split_val['user_data'][client] = Fair_Dataset(X[val_index,:], Y[val_index,:], A[val_index,:])
        split_test['user_data'][client] = Fair_Dataset(X[test_index,:], Y[test_index,:], A[test_index,:])

        split_train['num_samples'][client] = len(train_index)
        split_val['num_samples'][client] = len(val_index)
        split_test['num_samples'][client] = len(test_index)
        
    return split_train,split_val,split_test
    
def celeba_read_data(data_split, name=None, sensitive_process=None):
    split_train = {'users': [], 'user_data':{}, 'num_samples':{}}
    split_val = copy.deepcopy(split_train)
    split_test = copy.deepcopy(split_train)

    for client in data_split['users']:
        split_train['users'].append(client)
        split_val['users'].append(client)
        split_test['users'].append(client)

        X = np.array(data_split['user_data'][client]["x"]).astype(np.float32)

        Y = np.array(data_split['user_data'][client]["y"]).astype(np.float32).reshape(-1,1)

        A = np.array(data_split['user_data'][client]["A"]).astype(np.float32).reshape(-1,1)

        n = np.arange(X.shape[0])
        indices = np.random.permutation(n)
        train_index, val_index, test_index = indices[:int(len(n)*0.7)], indices[int(len(n)*0):int(len(n)*0.7)], indices[int(len(n)*0.4):]
        split_train['user_data'][client] = Fair_Dataset(X[train_index,:], Y[train_index,:], A[train_index,:])
        split_val['user_data'][client] = Fair_Dataset(X[val_index,:], Y[val_index,:], A[val_index,:])
        split_test['user_data'][client] = Fair_Dataset(X[test_index,:], Y[test_index,:], A[test_index,:])

        split_train['num_samples'][client] = len(train_index)
        split_val['num_samples'][client] = len(val_index)
        split_test['num_samples'][client] = len(test_index)
    
    return split_train,split_val,split_test