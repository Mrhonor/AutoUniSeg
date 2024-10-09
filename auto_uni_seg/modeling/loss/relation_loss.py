import torch
import torch.nn.functional as F
import numpy as np
import pickle
# def calculate_loss_numpy(M_A, M_B, Omega):
#     N, a = M_A.shape
#     _, b = M_B.shape
#     L_r = np.zeros((a, b))
#     for i in range(a):
#         for j in range(b):
#             if Omega[i, j] == 1:
#                 L_r[i, j] = 1 - np.max(M_A[:, i] * M_B[:, j])
#             else:
#                 L_r[i, j] = np.max(M_A[:, i] * M_B[:, j])
#     return L_r

def load_relation_gt(path):
    with open(path, "rb") as file:
        gt = pickle.load(file)
    return gt
    

def calculate_loss_torch(M_A, M_B, Omega):
    M_A = M_A.unsqueeze(2)  # [N, a, 1]
    M_B = M_B.unsqueeze(1)  # [N, 1, b]
    max_prod = torch.max(M_A * M_B, dim=0)[0]  # 得到 [a, b]
    loss = torch.mean(torch.where(Omega == 1, 1 - max_prod, max_prod))
    return loss

def relation_loss(M, dataset_cats, gt):
    adj_mI = M
    cur_cat = 0
    n_datasets = len(dataset_cats)
    
    out_adj = []
    for i in range(0, n_datasets):
        this_bipartite_graph = adj_mI[cur_cat:cur_cat+dataset_cats[i], :].T

        # softmax_bipartite_graph = F.softmax(this_bipartite_graph/0.07, dim=0)

        out_adj.append(this_bipartite_graph)
        cur_cat = cur_cat+dataset_cats[i]
        
    loss = torch.tensor(0.0, device='cuda')
    for i in range(0, n_datasets-1):
        for j in range(i+1, n_datasets):
            gt[i][j] = gt[i][j].cuda()
            
            loss += calculate_loss_torch(out_adj[i], out_adj[j], gt[i][j])
    return loss
            
    
    
    