import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.projects.point_rend.point_features import point_sample
import networkx as nx

class Edge:
    def __init__(self, to, next, flow, dis, from_):
        self.to = to  
        self.next = next  
        self.flow = flow  
        self.dis = dis  
        self.from_ = from_

class MinCostMaxFlow_NX(nn.Module):

    def __init__(self, n_classes, uni_classes, n_points=12544, ignore_lb=255):
        super(MinCostMaxFlow_NX, self).__init__()
        self.uni_classes = uni_classes
        n = n_classes + self.uni_classes + 2
        m = n_classes + self.uni_classes + 5 * n_classes 
        s = 0
        t = n-1


        self.n = n  
        self.m = m 
        self.s = s 
        self.t = t  

        self.num_points = n_points
        self.n_classes = n_classes
        
        self.G = nx.DiGraph()
        self.bceloss = torch.nn.BCEWithLogitsLoss(reduction='none') #torch.nn.BCELoss(reduction='none')
        self.ignore_lb = ignore_lb
        
    
    def construct_edges(self, unify_logits, target, bipart):
        bs = unify_logits.shape[0]
        # unify_logits = unify_logits
        # target = target
        # bipart = bipart

        point_coords = torch.rand(1, self.num_points, 2, device=unify_logits.device, dtype=unify_logits.dtype)
        # get gt labels
        tgt_mask = point_sample(
            target.unsqueeze(1).to(unify_logits.dtype),
            point_coords.repeat(target.shape[0], 1, 1),
            padding_mode='reflection',
            mode='nearest'
        ).squeeze(1)  #.view(-1)
        

        out_mask = point_sample(
            unify_logits,
            point_coords.repeat(target.shape[0], 1, 1),
            align_corners=False,
            padding_mode='reflection'
        )
        out_mask = F.softmax(out_mask, dim=1)
        
        self.link_num = 0
        self.total_links = 0
        for i in range(self.n_classes):
            tgt = tgt_mask == i
            if not tgt.any():
                continue
            tgt = tgt.to(unify_logits.dtype)
            losses = self.bceloss(out_mask.view(-1, self.num_points)[None], tgt.repeat(1, self.uni_classes, 1))
            
            costs = []
            for j in range(self.uni_classes):
                loss = torch.mean(losses[:, j, :])
                cost = (1-bipart[i, j]) + loss
                costs.append(cost)
            values, indexs = torch.tensor(costs).topk(5, largest=False)
            
            for idx, v in zip(indexs, values.cpu()):
                # self.G.add_edge(1, 2, weight=3)
                idx = int(idx)
                self.G.add_edge(1+idx, 1+self.uni_classes+i, capacity=1, weight=v)
                self.total_links += 1
                self.link_num += 1
                
                
        for j in range(self.uni_classes):
            self.G.add_edge(0, 1+j, capacity=1, weight=0)
            self.total_links += 1
        
        for i in range(self.n_classes):
            self.G.add_edge(1+self.uni_classes+i, self.n-1, capacity=2, weight=0)
            self.total_links += 1
                    

    def forward(self, unify_logits, target, bipart):
        self.construct_edges(unify_logits, target, bipart)

        solve = nx.max_flow_min_cost(self.G, 0, self.n-1)
        
        ret = 255 * torch.ones(self.uni_classes)
        for i in range(1, self.uni_classes+1):

            for k, v in solve[i].items():
                if v != 0:
                    ret[i-1] = k-1-self.uni_classes
                    break
        return ret
                
            

if __name__ == "__main__":
    # n, m, s, t = map(int, input().split())
    import time
    
    n_classes = 2
    unify_logits = torch.tensor([[[[1,1],
                                  [0,0]],
                                 [[1.5,0.1],
                                  [0,0]],
                                 [[1,1],
                                  [1,1]],
                                 [[0,0],
                                  [1,1]],
                                 [[0.2,1.5],
                                  [0,0]]]])
    target = torch.tensor([[[0,0],
                           [1,1]]])
    bipart = torch.tensor([[0.2,0.2,0.2,0.2,0.2],[0.2,0.2,0.2,0.2,0.2]])
    T1 = time.time()

    mcmf = MinCostMaxFlow_NX(2, 5,16, 255)
    


    print(mcmf(unify_logits, target, bipart))
    T2 = time.time()

