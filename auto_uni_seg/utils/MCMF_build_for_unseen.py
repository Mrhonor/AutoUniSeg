import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.projects.point_rend.point_features import point_sample
from ortools.graph import pywrapgraph
import logging

class Edge:
    def __init__(self, to, next, flow, dis, from_):
        self.to = to  
        self.next = next  
        self.flow = flow  
        self.dis = dis  
        self.from_ = from_

logger = logging.getLogger(__name__)

class MinCostMaxFlow_Mask(nn.Module):
    def __init__(self):
        super(MinCostMaxFlow_Mask, self).__init__()

        self.G = pywrapgraph.SimpleMinCostFlow()

        
    
    def construct_edges(self, cost_matrix):
        uni_class, target_class = cost_matrix.shape

        for i in range(uni_class):
            for j in range(target_class):

                self.G.AddArcWithCapacityAndUnitCost(1+i, 1+uni_class+j, 1, int(100*cost_matrix[uni_class][target_class]))
                                
        # print(self.link_num)
        for j in range(uni_class):
            # logger.info(f'add arc:{0}, {1+j}, {1}, {0}')
            self.G.AddArcWithCapacityAndUnitCost(0, 1+j, 1, 0)
        
        for i in range(target_class):
            if i == target_class-1:
                self.G.AddArcWithCapacityAndUnitCost(1+uni_class+i, uni_class+target_class+1, uni_class-target_class+1, 0)
            else:
                self.G.AddArcWithCapacityAndUnitCost(1+uni_class+i, uni_class+target_class+1, 3, 0)
        
        sup = uni_class
        # logger.info(max_sup_num)
        for i in range(uni_class+target_class+2):

            if i == 0:
                self.G.SetNodeSupply(i, sup)
            elif i == uni_class+target_class+1:
                self.G.SetNodeSupply(i, -sup+target_class-1)
            elif i < uni_class+target_class and i > uni_class:
                self.G.SetNodeSupply(i, -1)
            else:
                self.G.SetNodeSupply(i, 0)
                
                    

    def forward(self, cost_matrix):
        self.construct_edges(cost_matrix)
        uni_class, target_class = cost_matrix.shape
        status = self.G.Solve()
        if status == self.G.OPTIMAL:
        
            src = []
            tgt = []
            for arc in range(self.G.NumArcs()):
                # Can ignore arcs leading out of source or into sink.
                if self.G.Tail(arc) != 0 and self.G.Head(arc) != 1+uni_class+target_class and self.G.Head(arc) != uni_class+target_class:
                    # Arcs in the solution have a flow value of 1. Their start and end nodes
                    # give an assignment of worker to task.
                    if self.G.Flow(arc) > 0:
                        src.append(self.G.Tail(arc)-1)
                        tgt.append(self.G.Head(arc)-1-uni_class)
        else:
            logger.info("There was an issue with the min cost flow input.")
            logger.info(f"Status: {status}")
            raise Exception("error")
        return src, tgt
                
            


class MinCostMaxFlow_Unseen(nn.Module):

    def __init__(self):
        super(MinCostMaxFlow_Unseen, self).__init__()

        self.G = pywrapgraph.SimpleMinCostFlow()

        
    
    def construct_edges(self, cost_matrix):
        uni_class, target_class = cost_matrix.shape

        for i in range(uni_class):
            for j in range(target_class):

                self.G.AddArcWithCapacityAndUnitCost(1+i, 1+uni_class+j, 1, int(100*cost_matrix[i][j]))
                                
        # print(self.link_num)
        for j in range(uni_class):
            # logger.info(f'add arc:{0}, {1+j}, {1}, {0}')
            self.G.AddArcWithCapacityAndUnitCost(0, 1+j, 1, 0)
        
        for i in range(target_class):
            self.G.AddArcWithCapacityAndUnitCost(1+uni_class+i, uni_class+target_class+1, uni_class, 0)
        
        sup = uni_class
        # logger.info(max_sup_num)
        for i in range(uni_class+target_class+2):

            if i == 0:
                self.G.SetNodeSupply(i, sup)
            elif i == uni_class+target_class+1:
                self.G.SetNodeSupply(i, -sup+target_class)
            elif i < uni_class+target_class+1 and i > uni_class:
                self.G.SetNodeSupply(i, -1)
            else:
                self.G.SetNodeSupply(i, 0)
                
                    

    def forward(self, cost_matrix):
        self.construct_edges(cost_matrix)
        uni_class, target_class = cost_matrix.shape

        status = self.G.Solve()
        if status == self.G.OPTIMAL:
        
            src = []
            tgt = []
            for arc in range(self.G.NumArcs()):
                # Can ignore arcs leading out of source or into sink.
                if self.G.Tail(arc) != 0 and self.G.Head(arc) != 1+uni_class+target_class:
                    # Arcs in the solution have a flow value of 1. Their start and end nodes
                    # give an assignment of worker to task.
                    if self.G.Flow(arc) > 0:
                        src.append(self.G.Tail(arc)-1)
                        tgt.append(self.G.Head(arc)-1-uni_class)
        else:
            logger.info("There was an issue with the min cost flow input.")
            logger.info(f"Status: {status}")
            raise Exception("error")
        return src, tgt
                
            

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

    mcmf = MinCostMaxFlow_Or(2, 5, 16, 255)
    


    print(mcmf(unify_logits, target, bipart))
    T2 = time.time()


