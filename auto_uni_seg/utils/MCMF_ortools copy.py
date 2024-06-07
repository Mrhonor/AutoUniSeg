import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.projects.point_rend.point_features import point_sample
from ortools.graph.python import min_cost_flow
import logging

class Edge:
    """边结构体"""
    def __init__(self, to, next, flow, dis, from_):
        self.to = to  # 目标节点
        self.next = next  # 起点
        self.flow = flow  # 流量
        self.dis = dis  # 花费
        self.from_ = from_

logger = logging.getLogger(__name__)

class MinCostMaxFlow_Or(nn.Module):
    """最小费用最大流算法类"""
    def __init__(self, n_classes, uni_classes, n_points=12544, ignore_lb=255):
        super(MinCostMaxFlow_Or, self).__init__()
        self.uni_classes = uni_classes
        n = n_classes + self.uni_classes + 2
        m = n_classes + self.uni_classes + 5 * n_classes 
        s = 0
        t = n-1
        self.G = min_cost_flow.SimpleMinCostFlow()

        self.n = n  # 节点数
        self.m = m  # 边数
        self.s = s  # 源点
        self.t = t  # 汇点        

        self.num_points = n_points
        self.n_classes = n_classes
        
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
        
        max_sup_num = {}
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
                # logger.info(f'add arc:{1+idx}, {1+self.uni_classes+i}, {1}, {int(100*v)}')
                idx = int(idx)
                if idx not in max_sup_num:
                    max_sup_num[idx] = i
                self.G.add_arc_with_capacity_and_unit_cost(1+idx, 1+self.uni_classes+i, 1, int(100*v))
                self.total_links += 1
                
                
        # print(self.link_num)
        for j in range(self.uni_classes):
            # logger.info(f'add arc:{0}, {1+j}, {1}, {0}')
            self.G.add_arc_with_capacity_and_unit_cost(0, 1+j, 1, 0)
            self.total_links += 1
        
        for i in range(self.n_classes):
            # logger.info(f'add arc:{1+self.uni_classes+i}, {self.n-1}, {3}, {0}')
            self.G.add_arc_with_capacity_and_unit_cost(1+self.uni_classes+i, self.n-1, 3, 0)
            self.total_links += 1
        
        # sup = min(self.max_sup_num, 3*self.n_classes)
        # logger.info(max_sup_num)
        for i in range(self.n):

            if i == 0:
                self.G.set_node_supply(i, len(max_sup_num))
            elif i == self.n-1:
                self.G.set_node_supply(i, -len(max_sup_num))
            else:
                self.G.set_node_supply(i, 0)
                
                    

    def forward(self, unify_logits, target, bipart):
        self.construct_edges(unify_logits, target, bipart)
        """最小费用最大流算法"""
        status = self.G.SolveMaxFlowWithMinCost()
        if status == self.G.OPTIMAL:
        
            ret = 255 * torch.ones(self.uni_classes)
            for arc in range(self.G.num_arcs()):
                # Can ignore arcs leading out of source or into sink.
                if self.G.tail(arc) != 0 and self.G.head(arc) != self.n-1:
                    # Arcs in the solution have a flow value of 1. Their start and end nodes
                    # give an assignment of worker to task.
                    if self.G.flow(arc) > 0:
                        ret[self.G.tail(arc)-1] = self.G.head(arc) - 1 - self.uni_classes
        else:
            logger.info("There was an issue with the min cost flow input.")
            logger.info(f"Status: {status}, {self.n_classes}")
            raise Exception("error")
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
    # 初始化最小费用最大流算法类
    mcmf = MinCostMaxFlow_Or(2, 5, 16, 255)
    

    # 求解最小费用最大流
    print(mcmf(unify_logits, target, bipart))
    T2 = time.time()
    # 输出最大流量和最小花费
    # print(int(mcmf.maxflow), int(mcmf.mincost))
    # for e in mcmf.edge:
    #     print(f"edge {e.to}, dis: {e.dis}, flow: {e.flow}")
    print('程序运行时间:%s毫秒' % ((T2 - T1)*1000))
