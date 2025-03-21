import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from fedlearn.models.models import LRCalibModel
import torch.nn.functional as F
import copy
import cvxpy as cp
from netcal.scaling import BetaCalibration

class FFPClient(object):

    def __init__(self, cid, train_data, val_data, test_data, options={}, model=None):

        # load params
        self.cid = cid
        self.options = options
        self.local_lr = options['local_lr']
        self.wd = options['wd']
        self.sensitive_attr = options['sensitive_attr']
        self.batch_size = options['batch_size']
        self.data_info = options['data_info']
        self.client_num = options['num_users']

        # load data
        self.train_data, self.val_data ,self.test_data = train_data,val_data, test_data
        self.A = self.train_data.A
        self.batch_size = options['batch_size']
        self.fairness_constraints = options['fairness_constraints']
        self.fair_metric = self.fairness_constraints['fairness_measure']
        self.FFP_beta = options['FFP_beta']
        if self.fair_metric == 'DP':
            self.global_lamb = torch.tensor([0.1], requires_grad=False)
            self.local_mu = torch.tensor([0.5, 0.5], requires_grad=False)
        elif self.fair_metric == 'EO':
            self.global_lamb_1 = torch.tensor([0.1], requires_grad=False)
            self.global_lamb_2 = torch.tensor([0.1], requires_grad=False)
            self.local_mu_1 = torch.tensor([0.1, 0.1], requires_grad=False)
            self.local_mu_2 = torch.tensor([0.1, 0.1], requires_grad=False)
        self.post_local_round_mu = options['post_local_round_mu']
        self.post_local_round_lamb= options['post_local_round_lamb']
        self.post_lr = options['post_lr']
        self.calib = options['calibration']
        self.global_delta, self.local_delta = torch.tensor(self.fairness_constraints['global']), torch.tensor(self.fairness_constraints['local'])


        # initilaize local model
        self.model = model
        self.global_params = self.get_model_params()
        self.local_u = torch.zeros_like(self.global_params)
        self.global_fair_score = 0

        if options['criterion'] == 'celoss':
            self.criterion = nn.CrossEntropyLoss()
            self.mission = 'multiclass'
        elif options['criterion'] == 'mseloss':
            self.criterion = nn.MSELoss()
            self.mission = 'reg'
        elif options['criterion'] == 'bceloss':
            self.criterion = nn.BCELoss()
            self.mission = 'binary'
        self.num_local_round = options['num_local_round']

        # use gpu
        self.gpu = options['gpu'] if 'gpu' in options else False
        self.device = options['device']
        
        if 'gpu' in options and (options['gpu'] is True):
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(self.device)
            # print('>>> Use gpu on self.device {}'.format(self.device.index))

        train_size = len(self.train_data)
        val_size = len(self.val_data)
        test_size = len(self.test_data)
        
        self.train_dataloader = DataLoader(self.train_data, batch_size = self.batch_size, shuffle = True)
        self.train_dataloader_iter = enumerate(self.train_dataloader)
        self.val_dataloader = DataLoader(self.val_data, batch_size = self.batch_size , shuffle = False)
        self.test_dataloader = DataLoader(self.test_data, batch_size = self.batch_size, shuffle = False)

        self.train_samples_num = len(self.train_data)
        self.val_samples_num = len(self.val_data)
        self.test_samples_num = len(self.test_data)

        # optimizer
        if options['local_optimizer'].lower() == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.local_lr, weight_decay=self.wd)
        elif options['local_optimizer'].lower() == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.local_lr, weight_decay=self.wd)
        elif options['local_optimizer'].lower() == 'adagrad':
            self.optimizer = optim.adagrad(self.model.parameters(), lr=self.local_lr, weight_decay=self.wd)

        self.pi_0_c, self.pi_1_c = torch.tensor(self.data_info['val_client_A_info'][self.cid][0] / self.data_info['val_num']), torch.tensor(self.data_info['val_client_A_info'][self.cid][1] / self.data_info['val_num'])
        self.p_A_0, self.p_A_1 = torch.tensor(self.data_info['A_info'][0] / self.data_info['val_num']), torch.tensor(self.data_info['A_info'][1] / self.data_info['val_num'])
        self.N_0_c, self.N_1_c = torch.tensor(self.data_info['val_client_A_info'][self.cid][0]), torch.tensor(self.data_info['val_client_A_info'][self.cid][1])

        self.p_A0_Y1, self.p_A1_Y1 = torch.tensor(self.data_info['val_Y1_A_info'][0] / self.data_info['val_num']), torch.tensor(self.data_info['val_Y1_A_info'][1] / self.data_info['val_num'])
        self.p_A0_Y0, self.p_A1_Y0 = torch.tensor(self.data_info['val_Y0_A_info'][0] / self.data_info['val_num']), torch.tensor(self.data_info['val_Y0_A_info'][1] / self.data_info['val_num'])
        self.p_A0_Y1_c, self.p_A1_Y1_c = torch.tensor(self.data_info['val_client_Y1_A_info'][self.cid][0] / self.data_info['val_num']), torch.tensor(self.data_info['val_client_Y1_A_info'][self.cid][1] / self.data_info['val_num'])
        self.p_A0_Y0_c, self.p_A1_Y0_c = torch.tensor(self.data_info['val_client_Y0_A_info'][self.cid][0] / self.data_info['val_num']), torch.tensor(self.data_info['val_client_Y0_A_info'][self.cid][1] / self.data_info['val_num'])


    @staticmethod
    def move_model_to_gpu(model, options):
        if 'gpu' in options and (options['gpu'] is True):
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if 'device' not in options else options['device']
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.enabled = True
            # torch.backends.cudnn.benchmark = True
            model.to(device)
            print('>>> Use gpu on self.device {}'.format(device.index))
        else:
            print('>>> Do not use gpu')

    def set_params(self, flat_params):
        '''set model parameters, where input is a flat parameter'''
        self.model.set_params(flat_params)

    def get_model_params(self):
        '''get local flat model parameters, transform torch model parameters into flat tensor'''
        return self.model.get_flat_params()
    
    def get_global_params(self, global_params):
        self.global_params = copy.deepcopy(global_params)

    def get_next_train_batch(self, dataloader=None, dataloader_iter=None):
        if not dataloader:
            dataloader = self.train_dataloader
        if not dataloader_iter:
            dataloader_iter = self.train_dataloader_iter

        try:
            _, batch_data = dataloader_iter.__next__()
        except StopIteration:
            dataloader_iter = enumerate(dataloader)
            _, batch_data = dataloader_iter.__next__()

        return batch_data
    
    def get_val_score(self):

        val_dataloader_iter = enumerate(self.val_dataloader)

        val_score = torch.zeros_like(torch.tensor(self.val_data.Y))
        batch_start = 0

        with torch.no_grad(): 
            for i,  (x, y, a) in val_dataloader_iter:
                self.model.eval()
                if self.gpu:
                    x, y = torch.squeeze(x.to(self.device)), y.to(self.device)
                pred_score_batch = self.model(x).detach().clone().cpu()
                batch_size = pred_score_batch.size(0)
                val_score[batch_start:batch_start + batch_size] = pred_score_batch
                batch_start += batch_size
        
        assert len(val_score) == len(self.val_data)

        self.val_score = val_score.cpu().detach().clone()

        return self.val_score
    
    def calibration(self, train=True, test_score=None,group=None):
        if train == True:

            s_val_0_score = np.array(self.val_score[self.val_data.A == 0])
            s_val_1_score = np.array(self.val_score[self.val_data.A == 1])

            y_val_0 = np.array(self.val_data.Y[self.val_data.A == 0])
            y_val_1 = np.array(self.val_data.Y[self.val_data.A == 1])

            self.score_calibrated_0 = BetaCalibration()
            self.score_calibrated_1 = BetaCalibration()

            self.score_calibrated_0.fit(s_val_0_score, y_val_0)
            self.score_calibrated_1.fit(s_val_1_score, y_val_1)

            calibrated_score_0 = self.score_calibrated_0.transform(s_val_0_score)
            calibrated_score_1 = self.score_calibrated_1.transform(s_val_1_score)

            self.val_score[self.val_data.A == 0] = torch.tensor(calibrated_score_0, dtype=torch.float32)
            self.val_score[self.val_data.A == 1] = torch.tensor(calibrated_score_1, dtype=torch.float32)

            return True
        
        else:
            assert len(test_score.shape) == len(group.shape)

            s_val_0_score = np.array(test_score[group == 0])
            s_val_1_score = np.array(test_score[group == 1])

            calibrated_score_0 = self.score_calibrated_0.transform(s_val_0_score)
            calibrated_score_1 = self.score_calibrated_1.transform(s_val_1_score)

            test_score_calib = torch.zeros_like(test_score)
            test_score_calib[group==0] = torch.tensor(calibrated_score_0, dtype=torch.float32)
            test_score_calib[group==1] = torch.tensor(calibrated_score_1, dtype=torch.float32)

            return test_score_calib
    
    def r_beta(self, x, beta):
        return (1 / beta) * torch.log(1 + torch.exp(beta * x))
    
    def obj_H(self, mu, lamb):

        if self.fair_metric == 'DP':
            f_0 = 1 / self.N_0_c * self.r_beta(self.pi_0_c * (2 * self.val_score[self.val_data.A==0] - 1) - lamb * self.pi_0_c / self.p_A_0 - (mu[0] - mu[1]), self.FFP_beta)
            f_1 = 1 / self.N_1_c * self.r_beta(self.pi_1_c * (2 * self.val_score[self.val_data.A==1] - 1) + lamb * self.pi_1_c / self.p_A_1 + (mu[0] - mu[1]), self.FFP_beta)
            assert len(f_0) + len(f_1) == len(self.val_score)
            assert int(self.data_info['val_client_A_info'][self.cid][0]) + int(self.data_info['val_client_A_info'][self.cid][1]) == len(self.val_score)
            H = (torch.sum(f_0) + torch.sum(f_1)) + self.global_delta * torch.abs(lamb) / self.client_num + self.local_delta * (mu[0] + mu[1])

        elif self.fair_metric == 'EO':
            assert len(lamb) == 2
            assert len(mu) == 2
            f_0 = 1 / self.N_0_c * self.r_beta(self.pi_0_c * (2 * self.val_score[self.val_data.A==0] - 1) + lamb[0] * self.pi_0_c / self.p_A0_Y1 * self.val_score[self.val_data.A==0]
                                               + lamb[1] * self.pi_0_c / self.p_A0_Y0 * (1 - self.val_score[self.val_data.A==0])
                                               + (mu[0][0] - mu[0][1]) * self.pi_0_c / self.p_A0_Y1_c * self.val_score[self.val_data.A==0] + (mu[1][0] - mu[1][1]) * self.pi_0_c / self.p_A0_Y0_c * (1-self.val_score[self.val_data.A==0]), self.FFP_beta)
            f_1 = 1 / self.N_1_c * self.r_beta(self.pi_1_c * (2 * self.val_score[self.val_data.A==1] - 1) - lamb[0] * self.pi_1_c / self.p_A1_Y1 * self.val_score[self.val_data.A==1]
                                               - lamb[1] * self.pi_1_c / self.p_A1_Y0 * (1 - self.val_score[self.val_data.A==1])
                                               - (mu[0][0] - mu[0][1]) * self.pi_1_c / self.p_A1_Y1_c * self.val_score[self.val_data.A==1] - (mu[1][0] - mu[1][1]) * self.pi_1_c / self.p_A1_Y0_c * (1-self.val_score[self.val_data.A==1]), self.FFP_beta)
            assert len(f_0) + len(f_1) == len(self.val_score)
            assert int(self.data_info['val_client_A_info'][self.cid][0]) + int(self.data_info['val_client_A_info'][self.cid][1]) == len(self.val_score)
            H = (torch.sum(f_0) + torch.sum(f_1)) + self.global_delta * (torch.abs(lamb[0]) + torch.abs(lamb[1])) / self.client_num + self.local_delta * ((mu[0][0] + mu[0][1]) + ((mu[1][0] + mu[1][1])))

        return H
    
    def true_H(self, mu, lamb):

        if self.fairness_constraints['fairness_measure'] == 'DP':
            f_0 = 1 / self.N_0_c * torch.relu(self.pi_0_c * (2 * self.val_score[self.val_data.A==0] - 1) - lamb * self.pi_0_c / self.p_A_0 - (mu[0] - mu[1]))
            f_1 = 1 / self.N_1_c * torch.relu(self.pi_1_c * (2 * self.val_score[self.val_data.A==1] - 1) + lamb * self.pi_1_c / self.p_A_1 + (mu[0] - mu[1]))
            assert len(f_0) + len(f_1) == len(self.val_score)
            assert int(self.data_info['val_client_A_info'][self.cid][0]) + int(self.data_info['val_client_A_info'][self.cid][1]) == len(self.val_score)
            H = (torch.sum(f_0) + torch.sum(f_1)) + self.global_delta * torch.abs(lamb) / self.client_num + self.local_delta * (mu[0] + mu[1])

        elif self.fairness_constraints['fairness_measure'] == 'EO':
            f_0 = 1 / self.N_0_c * torch.relu(self.pi_0_c * (2 * self.val_score[self.val_data.A==0] - 1) + lamb[0] * self.pi_0_c / self.p_A0_Y1 * self.val_score[self.val_data.A==0] 
                                               + lamb[1] * self.pi_0_c / self.p_A0_Y0 * (1 - self.val_score[self.val_data.A==0]) 
                                               + (mu[0][0] - mu[0][1]) * self.pi_0_c / self.p_A0_Y1_c * self.val_score[self.val_data.A==0] + (mu[1][0] - mu[1][1]) * self.pi_0_c / self.p_A0_Y0_c * (1-self.val_score[self.val_data.A==0]))
            
            f_1 = 1 / self.N_1_c * torch.relu(self.pi_1_c * (2 * self.val_score[self.val_data.A==1] - 1) - lamb[0] * self.pi_1_c / self.p_A1_Y1 * self.val_score[self.val_data.A==1] 
                                               - lamb[1] * self.pi_1_c / self.p_A1_Y0 * (1 - self.val_score[self.val_data.A==1]) 
                                               - (mu[0][0] - mu[0][1]) * self.pi_1_c / self.p_A1_Y1_c * self.val_score[self.val_data.A==1] - (mu[1][0] - mu[1][1]) * self.pi_1_c / self.p_A1_Y0_c * (1-self.val_score[self.val_data.A==1]))
            assert len(f_0) + len(f_1) == len(self.val_score)
            assert int(self.data_info['val_client_A_info'][self.cid][0]) + int(self.data_info['val_client_A_info'][self.cid][1]) == len(self.val_score)
            H = (torch.sum(f_0) + torch.sum(f_1)) + self.global_delta * (torch.abs(lamb[0]) + torch.abs(lamb[1])) / self.client_num + self.local_delta * ((mu[0][0] + mu[0][1]) + ((mu[1][0] + mu[1][1])))

        return H

    def local_fair_post(self):

        if self.fair_metric == 'DP':
            self.local_mu_optim = self.local_mu.clone().detach()
            self.local_mu_optim.requires_grad_(True)

            for _ in range(self.post_local_round_mu):
                loss = self.obj_H(self.local_mu_optim, self.global_lamb)
                loss.backward()
                with torch.no_grad(): 
                    self.local_mu_optim *= torch.exp(-self.post_lr * self.local_mu_optim.grad)
                    self.local_mu_optim.clamp_(min=0)

            self.local_mu = self.local_mu_optim.detach().clone().requires_grad_(False)

            self.global_lamb_optim = self.global_lamb.clone().detach()
            self.global_lamb_optim.requires_grad_(True)

            for _ in range(self.post_local_round_lamb):
                loss = self.obj_H(self.local_mu, self.global_lamb_optim)
                loss.backward()
                with torch.no_grad(): 
                    self.global_lamb_optim -= self.post_lr * 5 * self.global_lamb_optim.grad
            return self.global_lamb_optim.detach().clone()
        
        elif self.fair_metric == 'EO':
            print(f"\n before train lamb: {self.global_lamb_1, self.global_lamb_2}, mu: {self.local_mu_1, self.local_mu_2}")
            self.local_mu_optim_1 = self.local_mu_1.clone().detach()
            self.local_mu_optim_2 = self.local_mu_2.clone().detach()
            

            self.local_mu_optim_1.requires_grad_(True)
            self.local_mu_optim_2.requires_grad_(False)
            for _ in range(self.post_local_round_mu):
                loss = self.obj_H([self.local_mu_optim_1, self.local_mu_optim_2], [self.global_lamb_1, self.global_lamb_2])
                loss.backward()
                with torch.no_grad():  
                    self.local_mu_optim_1 *= torch.exp(-self.post_lr * self.local_mu_optim_1.grad)
                    self.local_mu_optim_1.clamp_(min=0)
                self.local_mu_optim_1.grad.zero_()
            self.local_mu_optim_1.requires_grad_(False)
            self.local_mu_optim_2.requires_grad_(True)

            for _ in range(self.post_local_round_mu):
                loss = self.obj_H([self.local_mu_optim_1, self.local_mu_optim_2], [self.global_lamb_1, self.global_lamb_2])
                loss.backward()
                with torch.no_grad():  
                    self.local_mu_optim_2 *= torch.exp(-self.post_lr * self.local_mu_optim_1.grad)
                    self.local_mu_optim_2.clamp_(min=0)
                self.local_mu_optim_2.grad.zero_()

            self.local_mu_1 = self.local_mu_optim_1.detach().clone().requires_grad_(False)
            self.local_mu_2 = self.local_mu_optim_2.detach().clone().requires_grad_(False)

            self.global_lamb_optim_1 = self.global_lamb_1.clone().detach()
            self.global_lamb_optim_2 = self.global_lamb_2.clone().detach()

            
            self.global_lamb_optim_1.requires_grad_(True)
            self.global_lamb_optim_2.requires_grad_(False)

            for _ in range(self.post_local_round_lamb):
                loss = self.obj_H([self.local_mu_1, self.local_mu_2], [self.global_lamb_optim_1, self.global_lamb_optim_2])
                loss.backward()
                with torch.no_grad(): 
                    self.global_lamb_optim_1 -= self.post_lr * 5 * self.global_lamb_optim_1.grad
                    self.global_lamb_optim_1.clamp_(min=0)
                self.global_lamb_optim_1.grad.zero_()
            
            self.global_lamb_optim_1.requires_grad_(False)
            self.global_lamb_optim_2.requires_grad_(True)

            for _ in range(self.post_local_round_lamb):
                loss = self.obj_H([self.local_mu_1, self.local_mu_2], [self.global_lamb_optim_1, self.global_lamb_optim_2])
                loss.backward()
                with torch.no_grad(): 
                    self.global_lamb_optim_2 -= self.post_lr * 5 * self.global_lamb_optim_2.grad
                    self.global_lamb_optim_2.clamp_(min=0)
                self.global_lamb_optim_2.grad.zero_()

            return self.global_lamb_optim_1.detach().clone(), self.global_lamb_optim_2.detach().clone()

    
    def local_train(self):

        begin_time = time.time()

        for _ in range(self.num_local_round):
            self.model.train()
            (x, y, a) = self.get_next_train_batch()
            if self.gpu:
                x, y = torch.squeeze(x.to(self.device)), y.to(self.device).reshape(-1,1)
            self.optimizer.zero_grad()
            pred = self.model(x)
            loss = self.criterion(pred, y)
            loss.backward()
            self.optimizer.step() 

        self.local_model = self.get_model_params()

        train_stats = self.model_eval(self.train_dataloader)
        param_dict = {'norm': torch.norm(self.local_model).item(),
            'max': self.local_model.max().item(),
            'min': self.local_model.min().item()}
        
        return_dict = {'loss': train_stats['loss'] / train_stats['num'],
            'acc': train_stats['acc'] / train_stats['num']}
        return_dict.update(param_dict)

        end_time = time.time()
        stats = {'id': self.cid, 'time': round(end_time - begin_time, 2)}
        stats.update(return_dict)
        return (len(self.train_data), self.local_model), stats
    
    def obj_local_mu_EO(self,t_mu_1,t_mu_2):
        s0 = ((-self.global_lamb_1 / self.p_A0_Y1 
            - self.global_lamb_2 / self.p_A0_Y0 
            - (t_mu_1) / self.p_A0_Y1_c 
            - (t_mu_2) / self.p_A0_Y0_c) / 
            (2 + self.global_lamb_1 / self.p_A0_Y1 
            - self.global_lamb_2 / self.p_A0_Y0 
            + (t_mu_1) / self.p_A0_Y1_c 
            - (t_mu_2) / self.p_A0_Y0_c))
        s1 = ((self.global_lamb_1 / self.p_A1_Y1 
            + self.global_lamb_2 / self.p_A1_Y0 
            + (t_mu_1) / self.p_A1_Y1_c 
            + (t_mu_2) / self.p_A1_Y0_c) / 
            (2 - self.global_lamb_1 / self.p_A1_Y1 
            + self.global_lamb_2 / self.p_A1_Y0 
            - (t_mu_1) / self.p_A1_Y1_c 
            + (t_mu_2) / self.p_A1_Y0_c))
            
        thres_A0 = 0.5 + 0.5 * s0
        thres_A1 = 0.5 + 0.5 * s1

        max_adapt_1 = 0.01 * 2 * torch.min(self.p_A0_Y1_c , self.p_A1_Y1_c).item()
        max_adapt_2 = 0.01 * 2 * torch.min(self.p_A1_Y0_c, self.p_A0_Y0_c).item()
        max_iter = int(1 / max([max_adapt_1,max_adapt_2]))
        i=0

        disp_0 = -torch.sum(self.val_score[(self.val_data.A==0)*(self.val_data.Y==0)] >= thres_A0)/self.data_info['val_num'] / self.p_A0_Y0_c + torch.sum(self.val_score[(self.val_data.A==1)*(self.val_data.Y==0)] >= thres_A1) / self.data_info['val_num']/ self.p_A1_Y0_c
        disp_1 = torch.sum(self.val_score[(self.val_data.A==0)*(self.val_data.Y==1)] >= thres_A0) /self.data_info['val_num'] / self.p_A0_Y1_c - torch.sum(self.val_score[(self.val_data.A==1)*(self.val_data.Y==1)] >= thres_A1) /self.data_info['val_num'] / self.p_A1_Y1_c
        while disp_0.abs() > self.local_delta.abs()*2 or disp_1.abs() > self.local_delta.abs()*2 and i < max_iter:
            if disp_0.abs() > self.local_delta.abs():
                if disp_0 > self.local_delta:
                    t_mu_2 += max_adapt_2
                elif disp_0 < -self.local_delta:
                    t_mu_2 -= max_adapt_2
            if disp_1.abs() > self.local_delta.abs():
                if disp_1> self.local_delta:
                    t_mu_1 += max_adapt_1
                elif disp_1 < -self.local_delta:
                    t_mu_1 -= max_adapt_1
            
            s0 = ((-self.global_lamb_1 / self.p_A0_Y1 - self.global_lamb_2 / self.p_A0_Y0 - (t_mu_1) / self.p_A0_Y1_c - (t_mu_2) / self.p_A0_Y0_c) / 
            (2 + self.global_lamb_1 / self.p_A0_Y1 - self.global_lamb_2 / self.p_A0_Y0 + (t_mu_1) / self.p_A0_Y1_c - (t_mu_2) / self.p_A0_Y0_c))
            s1 = ((self.global_lamb_1 / self.p_A1_Y1 + self.global_lamb_2 / self.p_A1_Y0 + (t_mu_1) / self.p_A1_Y1_c + (t_mu_2) / self.p_A1_Y0_c) / 
            (2 - self.global_lamb_1 / self.p_A1_Y1 + self.global_lamb_2 / self.p_A1_Y0 - (t_mu_1) / self.p_A1_Y1_c + (t_mu_2) / self.p_A1_Y0_c))
            thres_A0 = 0.5 + 0.5 * s0
            thres_A1 = 0.5 + 0.5 * s1
            disp_0 = -torch.sum(self.val_score[(self.val_data.A==0)*(self.val_data.Y==0)] >= thres_A0)/self.data_info['val_num'] / self.p_A0_Y0_c + torch.sum(self.val_score[(self.val_data.A==1)*(self.val_data.Y==0)] >= thres_A1)/self.data_info['val_num'] / self.p_A1_Y0_c
            disp_1 = torch.sum(self.val_score[(self.val_data.A==0)*(self.val_data.Y==1)] >= thres_A0)/self.data_info['val_num'] / self.p_A0_Y1_c - torch.sum(self.val_score[(self.val_data.A==1)*(self.val_data.Y==1)] >= thres_A1)/self.data_info['val_num'] / self.p_A1_Y1_c
            i+=1

        print(f'client: {self.cid}, disp: {disp_0,disp_1}, sample_num:{int(self.N_0_c), int(self.N_1_c)}, Group-0 classifier threshold: {float(thres_A0):.4f}, Group-1 classifier threshold: {float(thres_A1):.4f}, Ps:{self.pi_0_c,self.pi_1_c},max_iter:{max_iter},i:{i}')
        return t_mu_1,t_mu_2
    
    def obj_local_mu(self, t_mu):

        mu_thres = t_mu

        thres_A0 = 0.5 + 0.5 / self.p_A_0 * self.global_lamb + 0.5 / self.pi_0_c * mu_thres
        thres_A1 = 0.5 - 0.5 / self.p_A_1 * self.global_lamb - 0.5 / self.pi_0_c * mu_thres

        disp = torch.sum(self.val_score[self.val_data.A==0] >= thres_A0) / self.N_0_c - torch.sum(self.val_score[self.val_data.A==1] >= thres_A1) / self.N_1_c

        i=0
        max_adapt = 0.001 * 2 * torch.min(self.pi_0_c, self.pi_1_c).item()
        max_iter = int(1 / max_adapt)
        while disp.abs() > self.local_delta.abs() and i < max_iter:
            if disp > self.local_delta:
                mu_thres += max_adapt
            elif disp < -self.local_delta:
                mu_thres -= max_adapt
            thres_A0 = 0.5 + 0.5 / self.p_A_0 * self.global_lamb + 0.5 / self.pi_0_c * mu_thres
            thres_A1 = 0.5 - 0.5 / self.p_A_1 * self.global_lamb - 0.5 / self.pi_1_c * mu_thres
            disp = torch.sum(self.val_score[self.val_data.A==0] >= thres_A0) / self.N_0_c - torch.sum(self.val_score[self.val_data.A==1] >= thres_A1) / self.N_1_c
            i+=1

        print(f'client: {self.cid}, disp: {disp}, sample_num:{int(self.N_0_c), int(self.N_1_c)}, Group-0 classifier threshold: {float(thres_A0):.4f}, Group-1 classifier threshold: {float(thres_A1):.4f}, Ps:{self.pi_0_c,self.pi_1_c}')
            
        return mu_thres
    
    def local_post_eval(self):
        if self.fair_metric == 'DP':
            self.local_mu_optim = self.local_mu.clone().detach()
            self.local_mu_optim.requires_grad_(True)

            for _ in range(self.post_local_round_mu*5):
                loss = self.true_H(self.local_mu_optim, self.global_lamb)
                loss.backward()
                with torch.no_grad():  # 更新参数时不追踪梯度
                    self.local_mu_optim *=torch.exp( -self.post_lr * self.local_mu_optim.grad)
                    self.local_mu_optim.clamp_(min=0)
                self.local_mu_optim.grad.zero_()

            self.local_mu = self.local_mu_optim.detach().clone().requires_grad_(False)
            thres_A0 = 0.5 + 0.5 / self.p_A_0 * (self.global_lamb) + 0.5 / self.pi_0_c * (self.local_mu[0] - self.local_mu[1])
            thres_A1 = 0.5 - 0.5 / self.p_A_1 * (self.global_lamb) - 0.5 / self.pi_1_c * (self.local_mu[0] - self.local_mu[1])

        elif self.fair_metric == 'EO':
            t_mu_1 = self.local_mu_1[0] - self.local_mu_1[1]
            t_mu_2 = self.local_mu_2[0] - self.local_mu_2[1]

            t_mu_1,t_mu_2 = self.obj_local_mu_EO(t_mu_1, t_mu_2)
            self.local_mu_1[0] = t_mu_1.abs() if t_mu_1 > 0 else 0
            self.local_mu_1[1] = 0 if t_mu_1 > 0 else t_mu_1.abs()
            self.local_mu_2[0] = t_mu_2.abs() if t_mu_2 > 0 else 0
            self.local_mu_2[1] = 0 if t_mu_2 > 0 else t_mu_2.abs()

            s0 = ((-self.global_lamb_1 / self.p_A0_Y1 - self.global_lamb_2 / self.p_A0_Y0 - (t_mu_1) / self.p_A0_Y1_c - (t_mu_2) / self.p_A0_Y0_c) / 
            (2 + self.global_lamb_1 / self.p_A0_Y1 - self.global_lamb_2 / self.p_A0_Y0 + (t_mu_1) / self.p_A0_Y1_c - (t_mu_2) / self.p_A0_Y0_c))
            s1 = ((self.global_lamb_1 / self.p_A1_Y1 + self.global_lamb_2 / self.p_A1_Y0 + (t_mu_1) / self.p_A1_Y1_c + (t_mu_2) / self.p_A1_Y0_c) / 
            (2 - self.global_lamb_1 / self.p_A1_Y1 + self.global_lamb_2 / self.p_A1_Y0 - (t_mu_1) / self.p_A1_Y1_c + (t_mu_2) / self.p_A1_Y0_c))
            thres_A0 = 0.5 + 0.5 * s0
            thres_A1 = 0.5 + 0.5 * s1

        print(f'\n client: {self.cid}, Group-0 classifier threshold: {float(thres_A0):.4f}, Group-1 classifier threshold: {float(thres_A1):.4f}')

        val_data_test = self.model_eval(self.val_dataloader,val=True, post_threshold=[thres_A0, thres_A1], local_fair=True, calibation=self.calib)
        test_data_test  = self.model_eval(self.test_dataloader,val=False, post_threshold=[thres_A0, thres_A1], local_fair=True, calibation=self.calib)
        return {'val': val_data_test,'test':test_data_test}

    def model_eval(self, dataLoader, val=False, post_threshold=None, local_fair=False, calibation=False):

        dataloader_iter = enumerate(dataLoader)

        self.model.eval()
        test_loss = test_acc = test_num = 0.0
        local_fair_info = {"pred_Y=1_A=0":0,"pred_Y=1_A=1":0, 
                           "pred_Y=1_A=0_Y=1":0,"pred_Y=1_A=1_Y=1":0, "pred_Y=1_A=0_Y=0":0,"pred_Y=1_A=1_Y=0":0,
                           "score_Y_A=0":0,"score_Y_A=1":0,
                           "score_Y_A=0_Y=1":0,"score_Y_A=1_Y=1":0, "score_Y_A=0_Y=0":0,"score_Y_A=1_Y=0":0,}
        if val == True:
            val_score = torch.zeros_like(torch.tensor(self.val_data.Y), dtype=torch.float32)
            batch_start = 0
        with torch.no_grad():
            for i, (x, y, A) in dataloader_iter:
                if self.gpu:
                    x, y, A = x.to(self.device), y.to(self.device).reshape(-1,1), A.to(self.device).reshape(-1,1)
                
                pred = self.model(x)

                if calibation == True:
                    pred = self.calibration(train=False, test_score=pred.detach().clone().cpu(), group=A.detach().clone().cpu()).to(self.device)

                if val==True:
                    pred_score_batch = pred.detach().clone().cpu()
                    batch_size = pred_score_batch.size(0)
                    val_score[batch_start:batch_start + batch_size] = pred_score_batch
                    batch_start += batch_size

                criterion = self.criterion

                loss = criterion(pred, y)

                if self.mission == 'binary':
                    if post_threshold is None:
                        predicted = (torch.sign(pred - 0.5) + 1) / 2
                    else:
                        shres = (1-A) * post_threshold[0].to(self.device) + A *post_threshold[1].to(self.device)
                        assert len(A) == len(shres)
                        predicted = (torch.sign(pred - shres) + 1) / 2
                elif self.mission == 'multiclass':
                    _, predicted = torch.max(pred, 1)
                    
                correct = predicted.eq(y).sum().item()
                batch_size = y.size(0)

                test_loss += loss.item() * y.size(0) # total loss, not average
                test_acc += correct # total acc, not average
                test_num += batch_size 
                local_fair_info['pred_Y=1_A=0'] += torch.sum((predicted==1) * (A==0)).cpu().detach()
                local_fair_info['pred_Y=1_A=1'] += torch.sum((predicted==1) * (A==1)).cpu().detach()
                local_fair_info['pred_Y=1_A=0_Y=1'] += torch.sum((predicted==1) * (A==0) * (y==1)).cpu().detach()
                local_fair_info['pred_Y=1_A=1_Y=1'] += torch.sum((predicted==1) * (A==1) * (y==1)).cpu().detach()
                local_fair_info['pred_Y=1_A=0_Y=0'] += torch.sum((predicted==1) * (A==0) * (y==0)).cpu().detach()
                local_fair_info['pred_Y=1_A=1_Y=0'] += torch.sum((predicted==1) * (A==1) * (y==0)).cpu().detach()
                local_fair_info['score_Y_A=0'] += torch.sum(pred[(A==0)]).cpu().detach().clone()
                local_fair_info['score_Y_A=1'] += torch.sum(pred[(A==1)]).cpu().detach().clone()
                local_fair_info['score_Y_A=0_Y=1'] += torch.sum(pred[(A==0) * (y==1)]).cpu().detach().clone()
                local_fair_info['score_Y_A=1_Y=1'] += torch.sum(pred[(A==1) * (y==1)]).cpu().detach().clone()
                local_fair_info['score_Y_A=0_Y=0'] += torch.sum(pred[(A==0) * (y==0)]).cpu().detach().clone()
                local_fair_info['score_Y_A=1_Y=0'] += torch.sum(pred[(A==1) * (y==0)]).cpu().detach().clone()

        test_dict = {'loss': test_loss, 'acc': test_acc, 'num': test_num, 'local_fair_info':local_fair_info}
        if local_fair == True:
            if self.fairness_constraints['fairness_measure'] == 'DP':
                dataset = 'val' if val else 'test'
                local_DP = torch.abs(local_fair_info['pred_Y=1_A=0'] / self.data_info[dataset+'_client_A_info'][self.cid][0] - local_fair_info['pred_Y=1_A=1'] / self.data_info[dataset+'_client_A_info'][self.cid][1])
                disp = torch.sum(self.val_score[self.val_data.A==0] >= post_threshold[0]) / self.N_0_c - torch.sum(self.val_score[self.val_data.A==1] >= post_threshold[1]) / self.N_1_c
                test_dict.update({'local_fair_measure':local_DP})
            if self.fairness_constraints['fairness_measure'] == 'EO':
                dataset = 'val' if val else 'test'
                local_EO_1 = torch.abs(local_fair_info['pred_Y=1_A=0_Y=1'] / self.data_info[dataset+'_client_Y1_A_info'][self.cid][0] - local_fair_info['pred_Y=1_A=1_Y=1'] / self.data_info[dataset+'_client_Y1_A_info'][self.cid][1])
                local_EO_0 = torch.abs(local_fair_info['pred_Y=1_A=0_Y=0'] / self.data_info[dataset+'_client_Y0_A_info'][self.cid][0] - local_fair_info['pred_Y=1_A=1_Y=0'] / self.data_info[dataset+'_client_Y0_A_info'][self.cid][1])
                local_EO = torch.max(local_EO_1,local_EO_0)
                test_dict.update({'local_fair_measure':local_EO})
        return test_dict

    def local_eval(self):
        train_data_test = self.model_eval(self.train_dataloader)
        val_data_test = self.model_eval(self.val_dataloader)
        test_data_test  = self.model_eval(self.test_dataloader)
        return {'train':train_data_test, 'val': val_data_test,'test':test_data_test}

        