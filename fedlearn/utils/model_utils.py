import torch
import numpy as np
import torch.nn.functional as F
import copy
import random

# data_info: {'train_num':train_num, 'test_num':test_num, 'Ylabel':Ylabel, 'Alabel':Alabel, 'A_num':A_num, 'Y_num':Y_num}

def global_EOD_1(split_data, model, device, data_info):
    fair_stats_A_Y0 = fair_stats_A_Y1 = np.zeros(data_info['A_num'])
    num_Y0 = num_Y1 = np.zeros(1)
    for sensitive_attr in (list(data_info['Alabel'])):
        for c_data in split_data['user_data'].values():
            fair_stats_A_Y1[int(sensitive_attr)] += np.sum((c_data.A == sensitive_attr) * (c_data.Y == 1))
            fair_stats_A_Y0[int(sensitive_attr)] += np.sum((c_data.A == sensitive_attr) * (c_data.Y == 0))
            num_Y0 += np.sum((c_data.Y == 0))
            num_Y1 += np.sum((c_data.Y == 1))


    local_A_Y1 = local_A_Y0 = np.zeros(data_info['A_num'])
    pred_1_Y1 = pred_1_Y0 = np.zeros(1)
    for c_data in split_data['user_data'].values():
        Y_score = torch.zeros((len(c_data),1))

        p = 512 # batch
        idxs = [list(range(i*512, (i+1)*512)) for i in range(len(c_data) // p)]
        idxs.append(list(range((len(c_data) // p) * p, len(c_data))))

        for idx in idxs:
            Y_score[idx] = model(torch.tensor(c_data.X[idx]).to(device)).detach().cpu()

        # Y_score = model(torch.tensor(c_data.X).to(device)).detach().cpu()
        Y_pred = ((torch.sign(Y_score - 0.5) + 1 ) / 2).numpy()
        for sensitive_attr in list(data_info['Alabel']):
            local_A_Y1[int(sensitive_attr)] += np.sum(Y_pred * (c_data.A == sensitive_attr) * (c_data.Y == 1))
            local_A_Y0[int(sensitive_attr)] += np.sum(Y_pred * (c_data.A == sensitive_attr) * (c_data.Y == 0))
            pred_1_Y1 += np.sum(Y_pred * (c_data.Y == 1))
            pred_1_Y0 += np.sum(Y_pred * (c_data.Y == 0))
    
    P_stat_0 = local_A_Y0 / fair_stats_A_Y0
    P_stat_1 = local_A_Y1 / fair_stats_A_Y1
    Y_stat_0 = pred_1_Y0 / num_Y0
    Y_stat_1 = pred_1_Y1 / num_Y1
    return (np.max([[np.abs(i-j) for i in P_stat_0] for j in P_stat_0]) + np.max([[np.abs(i-j) for i in P_stat_1] for j in P_stat_1])), (np.max(np.abs(P_stat_0 - Y_stat_0)) + np.max(np.abs(P_stat_1 - Y_stat_1)))

def global_EOD(split_data, model, device, data_info):
    assert data_info['A_num'] == 2
    fair_stats_A_Y0, fair_stats_A_Y1 = [0,0], [0,0] # |{A=a, Y=y}|
    num_Y0, num_Y1 = 0, 0# |{Y=y}|
    for sensitive_attr in (0,1):
        for c_data in split_data['user_data'].values():
            fair_stats_A_Y1[sensitive_attr] += np.sum((c_data.A == sensitive_attr) * (c_data.Y == 1))
            fair_stats_A_Y0[sensitive_attr] += np.sum((c_data.A == sensitive_attr) * (c_data.Y == 0))
            num_Y0 += np.sum((c_data.Y == 0))
            num_Y1 += np.sum((c_data.Y == 1))


    local_A_Y1, local_A_Y0 = [0,0], [0,0] # |{Y'=1, A=a, Y=y}|
    pred_1_Y1, pred_1_Y0 = 0, 0 # |{Y'=1, Y=y}|
    for c_data in split_data['user_data'].values():
        Y_score = torch.zeros((len(c_data),1))
        p = 512 # batch
        idxs = [list(range(i*p, (i+1)*p)) for i in range(len(c_data) // p)]
        idxs.append(list(range((len(c_data) // p) * p, len(c_data))))

        for idx in idxs:
            Y_score[idx] = model(torch.tensor(c_data.X[idx]).to(device)).detach().cpu()

        # Y_score = model(torch.tensor(c_data.X).to(device)).detach().cpu()
        Y_pred = ((torch.sign(Y_score - 0.5) + 1 ) / 2).numpy()
        for sensitive_attr in (0,1):
            local_A_Y1[sensitive_attr] += np.sum(Y_pred * (c_data.A == sensitive_attr) * (c_data.Y == 1))
            local_A_Y0[sensitive_attr] += np.sum(Y_pred * (c_data.A == sensitive_attr) * (c_data.Y == 0))
            pred_1_Y1 += np.sum(Y_pred * (c_data.Y == 1))
            pred_1_Y0 += np.sum(Y_pred * (c_data.Y == 0))
    
    P_stat_0 = [local_A_Y0[sensitive_attr] / fair_stats_A_Y0[sensitive_attr] for sensitive_attr in (0,1)]
    P_stat_1 = [local_A_Y1[sensitive_attr] / fair_stats_A_Y1[sensitive_attr] for sensitive_attr in (0,1)]
    Y_stat_0 = pred_1_Y0 / num_Y0
    Y_stat_1 = pred_1_Y1 / num_Y1
    return np.max([np.abs(P_stat_0[0] - P_stat_0[1]), np.abs(P_stat_1[0] - P_stat_1[1])]), np.abs(P_stat_0[0] - P_stat_0[1])

def global_EOD_score(split_data, model, device, data_info):
    assert data_info['A_num'] == 2
    fair_stats_A_Y0, fair_stats_A_Y1 = [0,0], [0,0] # |{A=a, Y=y}|
    num_Y0, num_Y1 = 0, 0# |{Y=y}|
    for sensitive_attr in (0,1):
        for c_data in split_data['user_data'].values():
            fair_stats_A_Y1[sensitive_attr] += np.sum((c_data.A == sensitive_attr) * (c_data.Y == 1))
            fair_stats_A_Y0[sensitive_attr] += np.sum((c_data.A == sensitive_attr) * (c_data.Y == 0))
            num_Y0 += np.sum((c_data.Y == 0))
            num_Y1 += np.sum((c_data.Y == 1))


    local_A_Y1, local_A_Y0 = [0,0], [0,0] # |{Y'=1, A=a, Y=y}|
    pred_1_Y1, pred_1_Y0 = 0, 0 # |{Y'=1, Y=y}|
    for c_data in split_data['user_data'].values():
        Y_score = torch.zeros((len(c_data),1))
        p = 512 # batch
        idxs = [list(range(i*p, (i+1)*p)) for i in range(len(c_data) // p)]
        idxs.append(list(range((len(c_data) // p) * p, len(c_data))))

        for idx in idxs:
            Y_score[idx] = model(torch.tensor(c_data.X[idx]).to(device)).detach().cpu()

        # Y_score = model(torch.tensor(c_data.X).to(device)).detach().cpu()
        # Y_pred = ((torch.sign(Y_score - 0.5) + 1 ) / 2).numpy()
        for sensitive_attr in (0,1):
            local_A_Y1[sensitive_attr] += np.sum(Y_score * (c_data.A == sensitive_attr) * (c_data.Y == 1))
            local_A_Y0[sensitive_attr] += np.sum(Y_score * (c_data.A == sensitive_attr) * (c_data.Y == 0))
            pred_1_Y1 += np.sum(Y_score * (c_data.Y == 1))
            pred_1_Y0 += np.sum(Y_score * (c_data.Y == 0))
    
    P_stat_0 = [local_A_Y0[sensitive_attr] / fair_stats_A_Y0[sensitive_attr] for sensitive_attr in (0,1)]
    P_stat_1 = [local_A_Y1[sensitive_attr] / fair_stats_A_Y1[sensitive_attr] for sensitive_attr in (0,1)]
    Y_stat_0 = pred_1_Y0 / num_Y0
    Y_stat_1 = pred_1_Y1 / num_Y1
    return (P_stat_0[0] - P_stat_0[1])**2 + (P_stat_1[0] - P_stat_1[1])**2, np.abs(P_stat_0[0] - P_stat_0[1])


# def global_EOD(split_data, model, device, data_info):
#     fair_stats_A = np.zeros(data_info['A_num'])
#     num_Y1 = np.zeros(1)
#     for sensitive_attr in list(data_info['Alabel']):
#         for c_data in split_data['user_data'].values():
#             fair_stats_A[int(sensitive_attr)] += np.sum((c_data.A == sensitive_attr) * (c_data.Y == 1))
#             num_Y1 += np.sum((c_data.Y == 1))

#     local_A = np.zeros(data_info['A_num'])
#     pred_Y1_Y1 = np.zeros(1)
#     for c_data in split_data['user_data'].values():
#         Y_score = torch.zeros((len(c_data),1))

#         p = 512 # batch
#         idxs = [list(range(i*512, (i+1)*512)) for i in range(len(c_data) // p)]
#         idxs.append(list(range((len(c_data) // p) * p, len(c_data))))

#         for idx in idxs:
#             Y_score[idx] = model(torch.tensor(c_data.X[idx]).to(device)).detach().cpu()

#         # Y_score = model(torch.tensor(c_data.X).to(device)).detach().cpu()
#         Y_pred = ((torch.sign(Y_score - 0.5) + 1 ) / 2).numpy()
#         for sensitive_attr in list(data_info['Alabel']):
#             local_A[int(sensitive_attr)] += np.sum(Y_pred * (c_data.A == sensitive_attr) * (c_data.Y == 1))
#             pred_Y1_Y1 += np.sum(Y_pred * (c_data.Y == 1))
    
#     P_stat = local_A / fair_stats_A
#     Y_stat = pred_Y1_Y1/num_Y1
#     return np.max([[np.abs(i-j) for i in P_stat] for j in P_stat]), np.max(np.abs(P_stat - Y_stat))

def global_DP(split_data, model, device, data_info):
    fair_stats_A = np.zeros(data_info['A_num'])
    data_num = np.zeros(1)
    for sensitive_attr in list(data_info['Alabel']):
        for c_data in split_data['user_data'].values():
            fair_stats_A[int(sensitive_attr)] += sum((c_data.A == sensitive_attr))
            data_num += len(c_data)
            # print(f"global_A_num: {sum((c_data.A == sensitive_attr))}")
    
    local_A = np.zeros(data_info['A_num'])
    pred_Y1 = np.zeros(1)
    for c_data in split_data['user_data'].values():
        Y_score = torch.zeros((len(c_data),1))

        p = 512
        idxs = [list(range(i*512, (i+1)*512)) for i in range(len(c_data) // p)]
        idxs.append(list(range((len(c_data) // p) * p, len(c_data))))

        for idx in idxs:
            Y_score[idx] = model(torch.tensor(c_data.X[idx]).to(device)).detach().cpu()

        # Y_score = model(torch.tensor(c_data.X).to(device)).detach().cpu()
        Y_pred = ((torch.sign(Y_score - 0.5) + 1 ) / 2).numpy()
        for sensitive_attr in list(data_info['Alabel']):
            local_A[int(sensitive_attr)] += np.sum(Y_pred * (c_data.A == sensitive_attr))
            pred_Y1 += np.sum(Y_pred)
            # print(f"local_A_num: {np.sum(Y_pred * (c_data.A == sensitive_attr))}")

    P_stat = local_A / fair_stats_A  
    Y_stat = pred_Y1/data_num
    # print(f"P_stat: {P_stat}")
    return np.max([[np.abs(i-j) for i in P_stat] for j in P_stat]), np.max(np.abs(P_stat - Y_stat))

def global_DP_score(split_data, model, device, data_info):
    assert data_info['A_num'] == 2
    fair_stats_A = np.zeros(data_info['A_num'])
    data_num = np.zeros(1)
    for sensitive_attr in (0,1):
        for c_data in split_data['user_data'].values():
            fair_stats_A[int(sensitive_attr)] += sum((c_data.A == sensitive_attr))
            data_num += len(c_data)
            # print(f"global_A_num: {sum((c_data.A == sensitive_attr))}")
    
    local_A = np.zeros(data_info['A_num'])
    pred_Y1 = np.zeros(1)
    for c_data in split_data['user_data'].values():
        Y_score = torch.zeros((len(c_data),1))

        p = 512
        idxs = [list(range(i*512, (i+1)*512)) for i in range(len(c_data) // p)]
        idxs.append(list(range((len(c_data) // p) * p, len(c_data))))

        for idx in idxs:
            Y_score[idx] = model(torch.tensor(c_data.X[idx]).to(device)).detach().cpu()

        # Y_score = model(torch.tensor(c_data.X).to(device)).detach().cpu()
        # Y_pred = ((torch.sign(Y_score - 0.5) + 1 ) / 2).numpy()
        for sensitive_attr in list(data_info['Alabel']):
            local_A[int(sensitive_attr)] += np.sum(Y_score * (c_data.A == sensitive_attr))
            pred_Y1 += np.sum(Y_score)
            # print(f"local_A_num: {np.sum(Y_pred * (c_data.A == sensitive_attr))}")

    P_stat = local_A / fair_stats_A
    Y_stat = pred_Y1/data_num
    # print(f"P_stat: {P_stat}")
    return P_stat[0] - P_stat[1]


# def global_DP(split_data, model, device, data_info):

#     fair_stats_A1 = fair_stats_A0 = 0
#     for c_data in split_data['user_data'].values():
#         fair_stats_A1 += sum(c_data.A)
#         fair_stats_A0 += sum((1 - c_data.A))
    
#     local_A0, local_A1 = 0, 0
#     for c_data in split_data['user_data'].values():
#         Y_score = model(torch.tensor(c_data.X).to(device)).detach().cpu()
#         Y_pred = (torch.sign(Y_score - 0.5) + 1 ) / 2
#         local_A0 += torch.sum(Y_pred * (1 - c_data.A))
#         local_A1 += torch.sum(Y_pred * (c_data.A))
    
#     P_stat_0 = local_A0  / fair_stats_A0
#     P_stat_1 = local_A1  / fair_stats_A1

#     return (P_stat_0 - P_stat_1).abs()

# def global_EOD(split_data, model, device, data_info):
#     fair_stats_A1 = fair_stats_A0 = 0
#     for c_data in split_data['user_data'].values():
#         fair_stats_A1 += sum(c_data.A * c_data.Y)
#         fair_stats_A0 += sum((1 - c_data.A) * c_data.Y)
    
#     local_A0, local_A1 = 0, 0
#     for c_data in split_data['user_data'].values():
#         Y_score = model(torch.tensor(c_data.X).to(device)).detach().cpu()
#         Y_pred = (torch.sign(Y_score - 0.5) + 1 ) / 2
#         local_A0 += torch.sum(Y_pred * (1 - c_data.A) * c_data.Y)
#         local_A1 += torch.sum(Y_pred * (c_data.A) * c_data.Y)
    
#     P_stat_0 = local_A0  / fair_stats_A0
#     P_stat_1 = local_A1  / fair_stats_A1

#     return (P_stat_0 - P_stat_1).abs()

def aggregate(wsolns): 
    
    total_weight = 0.0
    base = torch.zeros(len(wsolns[0][1]))

    for (w, soln) in wsolns:  # w is the number of samples

        total_weight += w 
        base += w * soln

    averaged_soln = base / total_weight

    return averaged_soln

def get_sort_idxs(vec:np.array):
    assert len(vec.shape) == 1
    idxs = np.arange(vec.shape[0])
    idxs_vec = np.vstack((vec, idxs))
    idxs_vec = idxs_vec[:, idxs_vec[0, :].argsort()]
    return idxs_vec[1,:]

def weighted_loss(logits, targets, weights, mean = True):
    acc_loss = F.binary_cross_entropy(logits, targets, reduction = 'none')
    if mean:
        weights_sum = weights.sum().item()
        acc_loss = torch.sum(acc_loss * weights / weights_sum)
    else:
        acc_loss = torch.sum(acc_loss * weights)
    return acc_loss

def get_cdf(data:torch.tensor, w=100):
    assert len(data.shape) == 1
    d = torch.linspace(0,1,w+1)
    value, index = torch.sort(data)
    f = torch.zeros_like(d)
    i = 0
    for k in value:
        while k >= (d[i]+1/w/2):
            i += 1
        f[i] += 1
    return torch.cumsum(f,dim=-1) / len(data)

def get_sample_target(distribution, num_samples):
    # distribution is the cdf 
    # num_samples is the number of samples to match
    cdf = copy.deepcopy(distribution)
    dsort = []
    eps = 1e-6
    j = 0
    for i in range(num_samples):
        p_right = (i + 1) * 1/num_samples
        # print(max(cdf - p_right))
        # target_right = min(torch.where(cdf - p_right > -eps)[0])
        while cdf[j] - p_right < - eps:
            j += 1
        target_right = j
        dsort.append(target_right)
    
    return torch.tensor(dsort)


# project the updated w to the probability simplex
def Proj_Weights_To_Simplex(w):
        
    # step 1: w is a list
    # create a copy of the list for sorting
    w = list(w)
    w_sorted = copy.copy(list(w))

    # step 2 i): sort w_(i) in ascending order
    w_sorted.sort()

    # step 2 ii): get n and i
    n = len(w_sorted) # n is the length of w
    i = n - 1 # i = n-1

    # step 3: this is a while loop on i
    while i > 0:
    
        # step 3 i): compute t_i
        t_i = 0
    
        # j take values on i, i+1, ..., n-1
        for j in range(i, n):
            # in python list, they are w_(i+1), w_(i+2), ..., w_(n)
            t_i += w_sorted[j]

        t_i -= 1
        t_i /= (n-i) # get t_i
    
        # step 3 ii): if t_i >= w_(i), then t_hat = t_i and go to step 5
        if t_i >= w_sorted[i-1]:
            t_hat = t_i
        
            # step 5: get x = (w-t_hat)_{+} and return x
            x = [0] * n

            for idx in range(n):
                x[idx] = max(w[idx]-t_hat, 0) # x = (w-t_hat)_{+}
            
            # return x as the output
            return x
    
        # else, if i >= 1, return to step 3
        else:
            i = i - 1

    # if i == 0, go to step 4
    # step 4: compute t_hat
    t_hat = 0

    for idx in range(n):
        t_hat += w[idx]

    t_hat -= 1
    t_hat /= n # get t_hat

    # step 5: get x = (w-t_hat)_{+} and return x
    x = [0] * n

    for idx in range(n):
        x[idx] = max(w[idx]-t_hat, 0) # x = (w-t_hat)_{+}

    # return x as the output
    return x

def Proj_Weights_To_Capped_Simplex(w, w_ub): 
        
        w = w.numpy()
        lb = np.zeros_like(w)
        ub = np.ones_like(2) * w_ub

        n = w.size #  n is the length of w
        total = np.sum(lb)

        lambdas = np.append(lb-w, ub-w)
        idx = np.argsort(lambdas)
        lambdas = lambdas[idx]

        active = 1
        for i in range(1, 2*n):
            total += active*(lambdas[i] - lambdas[i-1])

            if total >= 1:
                lam = (1-total) / active + lambdas[i]

                x = np.clip(w + lam, lb, ub)

                return x

            elif idx[i] < n:
                active += 1
            else:
                active -= 1

def set_seed(options, seed):
    np.random.seed(1 + seed)
    torch.manual_seed(12 + seed)
    random.seed(seed)
    if options['gpu']:
        torch.cuda.manual_seed_all(123 + seed)