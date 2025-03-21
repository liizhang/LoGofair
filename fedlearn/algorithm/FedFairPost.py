from fedlearn.algorithm.FedBase import Server as BasicServer
from fedlearn.models.FedFairPostClient import FFPClient
from fedlearn.utils.metric import Metrics
from tqdm import tqdm
import time
import numpy as np
import torch
import copy
import os

class Server(BasicServer):
    def __init__(self, dataset, options, name=''):
        super().__init__(dataset, options, name)
        self.clients = self.setup_clients(FFPClient, dataset, options)
        assert len(self.clients) > 0
        print('>>> Initialize {} clients in total'.format(len(self.clients)))

        # Initialize system metrics
        self.name = '_'.join([name, f'wn{self.clients_per_round}', f'tn{len(self.clients)}'])
        self.metrics = Metrics(self.clients, options)
        self.aggr = options['aggr']
        self.beta = options['beta']
        self.simple_average = options['simple_average']
        self.post_round = options['post_round']
        self.data_exist = options.get('data_exist',False)
        self.save_path = options['data_save_path'] 
        self.calibration = options['calibration']
        self.fair_info = options['fairness_constraints']
        self.fair_metric = self.fair_info['fairness_measure']
        self.fair_save = options['fair_save']
        if self.fair_metric == 'DP':
            self.global_lamb = torch.tensor([0.01], requires_grad=False)
        elif self.fair_metric == 'EO':
            self.global_lamb_1 = torch.tensor([0.01], requires_grad=False)
            self.global_lamb_2 = torch.tensor([0.01], requires_grad=False)

    def train(self):
        print('>>> Select {} clients for aggregation per round \n'.format(self.clients_per_round))

        pre_score_path = self.save_path + 'pre_score.pth'
        pre_model_path = self.save_path + 'pre_model.pth'

        if self.data_exist == True and os.path.exists(pre_model_path):
            self.latest_model = torch.load(pre_model_path)
            for c in self.clients:
                c.set_params(self.latest_model)
                c.get_val_score()
            
        else:
            if os.path.exists(pre_score_path):
                os.remove(pre_score_path)
            if os.path.exists(pre_model_path):
                os.remove(pre_model_path)
            
            # pre-trian the model
            if self.gpu:
                self.latest_model = self.latest_model.to(self.device)

            for self.current_round in tqdm(range(self.num_round)):
                tqdm.write('>>> Pre-trian Round {}, latest model.norm = {}'.format(self.current_round, self.latest_model.norm()))

                # Test latest model on train and eval data
                stats = self.test(self.clients, self.current_round)

                self.iterate()
                self.current_round += 1

            # Test final model on data
            self.stats = self.test(self.clients, self.current_round)

            # Save tracked information
            self.metrics.write()

            pre_score = {}
            torch.save(self.latest_model, pre_model_path)

            for c in self.clients:
                pre_score[c.cid] = c.get_val_score()

        if self.calibration == True:
            for c in self.clients:
                c.calibration(train=True)

        for self.current_round in tqdm(range(self.post_round)):
            tqdm.write('>>> Post round {}'.format(self.current_round))

            if self.fair_metric == 'DP':

                lamb_optim = []
                for c in self.clients:
                    # Set params
                    c.global_lamb = self.global_lamb
                    # Local training
                    local_lamb_optim = c.local_fair_post()
                    lamb_optim.append(local_lamb_optim)
            
                self.global_lamb = (sum(lamb_optim) / self.num_users)
                self.global_lamb.requires_grad_(False)

                # Test latest model on val and eval data
                stats = self.post_test(self.clients, self.current_round)

                print(f'\n global lamb:{self.global_lamb}.')

            elif self.fair_metric == 'EO':
                lamb_optim_1,lamb_optim_2 = [],[]
                for c in self.clients:
                    #Set params
                    c.global_lamb_1 = self.global_lamb_1
                    c.global_lamb_2 = self.global_lamb_2
                    local_lamb_optim_1,local_lamb_optim_2 = c.local_fair_post()
                    lamb_optim_1.append(local_lamb_optim_1),lamb_optim_2.append(local_lamb_optim_2)

                self.global_lamb_1 = (sum(lamb_optim_1) / self.num_users)
                self.global_lamb_1.requires_grad_(False)
                self.global_lamb_2 = (sum(lamb_optim_2) / self.num_users)
                self.global_lamb_2.requires_grad_(False)
                # Test latest model on val and eval data
                stats = self.post_test(self.clients, self.current_round)

                print(f'\n global lamb_1:{self.global_lamb_1}, global lamb_2:{self.global_lamb_2}.')

        if self.fair_save == True:
            result_save_path = self.save_path + 'parato_result.txt'
            with open(result_save_path, 'a') as file:
                training_results = f"=== fair metric: {self.fair_info['fairness_measure']}, acc: {float(stats['test_acc'])}, global :{float(stats['test_fairness'][self.fair_info['fairness_measure']])}, avg_local: {float(stats['test_avg_local_fair'])}"
                file.write("\n" + training_results)
                print(f"Training results appended to file.")



    def iterate(self):
        selected_clients = self.select_clients(self.clients, self.current_round)

        # Do local update for the selected clients
        solns, stats = [], []
        for c in selected_clients:
            # Communicate the latest global model
            c.set_params(self.latest_model)

            # Solve local and personal minimization
            soln, stat = c.local_train()
            
            if self.print:
                tqdm.write('>>> Round: {: >2d} local acc | CID:{}| loss {:>.4f} | Acc {:>5.2f}% | Time: {:>.2f}s'.format(
                    self.current_round, c.cid, stat['loss'], stat['acc'] * 100, stat['time']
                    ))
                        
            # Add solutions and stats
            solns.append(soln)
            stats.append(stat)

    
        self.latest_model = self.aggregate(solns, seed = self.current_round, stats = stats)
        return True

    def aggregate(self, solns, seed, stats):
        averaged_solution = torch.zeros_like(self.latest_model)

        num_samples, chosen_solns = [info[0] for info in solns], [info[1] for info in solns]
        if self.aggr == 'mean':  
            if self.simple_average:
                num = 0
                for num_sample, local_soln in zip(num_samples, chosen_solns):
                    num += 1
                    averaged_solution += local_soln
                averaged_solution /= num
            else:      
                selected_sample = 0
                for num_sample, local_soln in zip(num_samples, chosen_solns):
                    # print(num_sample)
                    averaged_solution += num_sample * local_soln
                    selected_sample += num_sample
                    # print("local_soln:{},num_sample:{}".format(local_soln, num_sample))
                averaged_solution = averaged_solution / selected_sample
                # print(averaged_solution)
        return averaged_solution.detach()
    
    def local_test(self, clients):
        assert self.latest_model is not None

        stats_dict = {'num_samples':[], 'accs':[], 'losses':[],'local_fair_info':[]}
        client_stats= {'train':copy.deepcopy(stats_dict), 
                       'val':copy.deepcopy(stats_dict),
                       'test' :copy.deepcopy(stats_dict)}

        for c in clients:
            if self.test_local:
                c.set_params(c.local_params)
            else:
                c.set_params(self.latest_model)
            client_test_dict = c.local_eval()

            for dataset, test_dict in client_test_dict.items():
                # dataset is train and test
                client_stats[dataset]['num_samples'].append(test_dict['num'])
                client_stats[dataset]['accs'].append(test_dict['acc'])
                client_stats[dataset]['losses'].append(test_dict['loss'])
                client_stats[dataset]['local_fair_info'].append(test_dict['local_fair_info'])
            
        ids = [c.cid for c in clients]

        return client_stats, ids
    
    def post_test(self, clients, current_round):
        begin_time = time.time()
        stats_dict = {'num_samples':[], 'accs':[],'local_fair_info':[],'local_fair_measure':[]}
        client_stats= {'val':copy.deepcopy(stats_dict),
                       'test' :copy.deepcopy(stats_dict)}
        for c in clients:
            if self.fair_metric == 'DP':
                c.global_lamb = self.global_lamb
            elif self.fair_metric == 'EO':
                c.global_lamb_1 = self.global_lamb_1
                c.global_lamb_2 = self.global_lamb_2
            client_eval_dict = c.local_post_eval()
            for dataset, test_dict in client_eval_dict.items():
                # dataset is train and test
                client_stats[dataset]['num_samples'].append(test_dict['num'])
                client_stats[dataset]['accs'].append(test_dict['acc'])
                client_stats[dataset]['local_fair_info'].append(test_dict['local_fair_info'])
                client_stats[dataset]['local_fair_measure'].append(test_dict['local_fair_measure'])

        end_time = time.time()

        val_acc = sum(client_stats['val']['accs']) / sum(client_stats['val']['num_samples'])
        val_fairness = self.fair_cal(client_stats, "val")
        
        test_acc = sum(client_stats['test']['accs']) / sum(client_stats['test']['num_samples'])
        test_fairness = self.fair_cal(client_stats, "test")

        local_fair = [float(client_stats['test']['local_fair_measure'][c]) for c in range(self.num_users)]
        avg_local_fair = sum(local_fair)/len(local_fair)

        model_stats = {'val_acc':val_acc, 'val_fairness':val_fairness,
                        'test_acc':test_acc, 'test_fairness':test_fairness, 'test_avg_local_fair':avg_local_fair,
                        'time': end_time - begin_time}
        
        self.post_print_result(current_round, model_stats, client_stats)

        return model_stats
    
    def post_print_result(self, current_round, model_stats, client_stats):
        tqdm.write("\n >>> Round: {: >4d} / Acc: {:.3%} / Time: {:.2f}s /  Fairness(val): global {}: {} ".format(
            current_round, model_stats['val_acc'], model_stats['time'], self.fair_metric, model_stats['val_fairness'][self.fair_metric]))
        
        for c in range(self.num_users):
            tqdm.write('>>> Client: {} / Acc: {:.3%} / Fair:{:.4f}'.format(
                        self.val_data['users'][c], 
                        client_stats['val']['accs'][c] / client_stats['val']['num_samples'][c], 
                        float(client_stats['val']['local_fair_measure'][c])
                      ))
        tqdm.write('=' * 102 + "\n")

        if current_round % self.eval_round == 0:
            tqdm.write("\n = Test = round: {} / acc: {:.3%} / Fairness(test) global {}: {} ".format(
                current_round, model_stats['test_acc'], self.fair_metric, model_stats['test_fairness'][self.fair_metric]))
            for c in range(self.num_users):
                tqdm.write('=== Test  Client: {} / Acc: {:.3%} / Fair(val):{:.4f} / Fair(test):{:.4f}'.format(
                            self.test_data['users'][c], 
                            client_stats['test']['accs'][c] / client_stats['test']['num_samples'][c], 
                            float(client_stats['val']['local_fair_measure'][c]),
                            float(client_stats['test']['local_fair_measure'][c])
                        ))
            local_fair = [float(client_stats['test']['local_fair_measure'][c]) for c in range(self.num_users)]
            tqdm.write(f"=== Test local fair {self.fair_metric}: {max(local_fair)}")
    
    def test(self, clients, current_round):
        begin_time = time.time()
        client_stats, ids = self.local_test(clients)
        end_time = time.time()

        train_acc = sum(client_stats['train']['accs']) / sum(client_stats['train']['num_samples'])
        train_loss = sum(client_stats['train']['losses']) / sum(client_stats['train']['num_samples'])

        train_fairness = self.fair_cal(client_stats, "train")

        val_acc = sum(client_stats['val']['accs']) / sum(client_stats['val']['num_samples'])
        val_loss = sum(client_stats['val']['losses']) / sum(client_stats['val']['num_samples'])

        val_fairness = self.fair_cal(client_stats, "val")
        
        test_acc = sum(client_stats['test']['accs']) / sum(client_stats['test']['num_samples'])
        test_loss = sum(client_stats['test']['losses']) / sum(client_stats['test']['num_samples'])

        test_fairness = self.fair_cal(client_stats, "test")

        model_stats = {'train_loss':train_loss, 'train_acc':train_acc, 'train_fairness':train_fairness,
                       'val_loss':val_loss, 'val_acc':val_acc, 'val_fairness':val_fairness,
                        'test_acc':test_acc, 'test_loss':test_loss, 'test_fairness':test_fairness,
                        'time': end_time - begin_time}
        
        self.print_result(current_round, model_stats, client_stats)

        return model_stats
    
    def fair_cal(self, client_stats, sets="train"):
        num_A0 = sum([self.data_info[sets+'_client_A_info'][c][0] for c in self.data_info[sets+'_client_A_info']])
        num_A1 = sum([self.data_info[sets+'_client_A_info'][c][1] for c in self.data_info[sets+'_client_A_info']])
        num_Y1_A0 = sum([self.data_info[sets+'_client_Y1_A_info'][c][0] for c in self.data_info[sets+'_client_Y1_A_info']])
        num_Y1_A1 = sum([self.data_info[sets+'_client_Y1_A_info'][c][1] for c in self.data_info[sets+'_client_Y1_A_info']])
        num_Y0_A0 = sum([self.data_info[sets+'_client_Y0_A_info'][c][0] for c in self.data_info[sets+'_client_Y0_A_info']])
        num_Y0_A1 = sum([self.data_info[sets+'_client_Y0_A_info'][c][1] for c in self.data_info[sets+'_client_Y0_A_info']])

        pred_Y1_A0 = pred_Y1_A1 = pred_Y1_A0_Y1 = pred_Y1_A1_Y1 = pred_Y1_A0_Y0 = pred_Y1_A1_Y0 = 0
        for c_fair_info in client_stats[sets]['local_fair_info']:
            pred_Y1_A0 += c_fair_info['pred_Y=1_A=0']
            pred_Y1_A1 += c_fair_info['pred_Y=1_A=1']
            pred_Y1_A0_Y1 += c_fair_info['pred_Y=1_A=0_Y=1']
            pred_Y1_A1_Y1 += c_fair_info['pred_Y=1_A=1_Y=1']
            pred_Y1_A0_Y0 += c_fair_info['pred_Y=1_A=0_Y=0']
            pred_Y1_A1_Y0 += c_fair_info['pred_Y=1_A=1_Y=0']

        score_Y_A0 = score_Y_A1 = score_Y_A0_Y1 = score_Y_A1_Y1 = score_Y_A0_Y0 = score_Y_A1_Y0 = 0
        for c_fair_info in client_stats[sets]['local_fair_info']:
            score_Y_A0 += c_fair_info['score_Y_A=0']
            score_Y_A1 += c_fair_info['score_Y_A=1']
            score_Y_A0_Y1 += c_fair_info['score_Y_A=0_Y=1']
            score_Y_A1_Y1 += c_fair_info['score_Y_A=1_Y=1']
            score_Y_A0_Y0 += c_fair_info['score_Y_A=0_Y=0']
            score_Y_A1_Y0 += c_fair_info['score_Y_A=1_Y=0']

        DP = pred_Y1_A0 / num_A0 - pred_Y1_A1 / num_A1
        cal_DP = torch.abs(torch.tensor(DP)).numpy()

        EO_Y1 = pred_Y1_A0_Y1 / num_Y1_A0 - pred_Y1_A1_Y1 / num_Y1_A1
        EO_Y0 = pred_Y1_A0_Y0 / num_Y0_A0 - pred_Y1_A1_Y0 / num_Y0_A1
        cal_EO = torch.max(torch.abs(torch.tensor(EO_Y1)),torch.abs(torch.tensor(EO_Y0))).numpy() 

        return {"DP":float(cal_DP), "EO":float(cal_EO)}
    
    def print_result(self, current_round, model_stats, client_stats):
        tqdm.write("\n >>> Round: {: >4d} / Acc: {:.3%} / Loss: {:.4f} / Time: {:.2f}s /  Fairness: {} ".format(
            current_round, model_stats['train_acc'], model_stats['train_loss'], model_stats['time'], model_stats['train_fairness']))
        
        for c in range(self.num_users):
            tqdm.write('>>> Client: {} / Acc: {:.3%} / Loss: {:.4f}'.format(
                        self.train_data['users'][c], 
                        client_stats['train']['accs'][c] / client_stats['train']['num_samples'][c], 
                        client_stats['train']['losses'][c] / client_stats['train']['num_samples'][c]
                      ))
        tqdm.write('=' * 102 + "\n")

        if current_round % self.eval_round == 0:
            tqdm.write("\n = Test = round: {} / acc: {:.3%} / loss: {:.4f} / {}: {} ".format(
                current_round, model_stats['test_acc'],model_stats['test_loss'], self.fair_metric, model_stats['test_fairness'][self.fair_metric]))