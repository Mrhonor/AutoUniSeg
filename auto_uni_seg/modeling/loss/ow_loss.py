import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

class OWLoss(nn.Module):
    def __init__(self, n_classes, hinged=False, delta=0.1):
        super().__init__()
        self.n_classes = n_classes
        self.hinged = hinged
        self.delta = delta
        self.count = torch.zeros(self.n_classes).cuda()  # count for class
        self.features = {
            i: torch.zeros(self.n_classes).cuda() for i in range(self.n_classes)
        }
        # See https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        # for implementation of Welford Alg.
        self.ex = {i: torch.zeros(self.n_classes).cuda() for i in range(self.n_classes)}
        self.ex2 = {
            i: torch.zeros(self.n_classes).cuda() for i in range(self.n_classes)
        }
        self.var = {
            i: torch.zeros(self.n_classes).cuda() for i in range(self.n_classes)
        }

        self.criterion = torch.nn.L1Loss(reduction="none")

        self.previous_features = None
        self.previous_count = None

    @torch.no_grad()
    def cumulate(self, logits: torch.Tensor, sem_gt: torch.Tensor):
        sem_pred = torch.argmax(torch.softmax(logits, dim=1), dim=1)
        gt_labels = torch.unique(sem_gt).tolist()
        logits_permuted = logits.permute(0, 2, 3, 1)
        for label in gt_labels:
            if label == 255:
                continue
            sem_gt_current = sem_gt == label
            sem_pred_current = sem_pred == label
            tps_current = torch.logical_and(sem_gt_current, sem_pred_current)
            if tps_current.sum() == 0:
                continue
            logits_tps = logits_permuted[torch.where(tps_current == 1)]
            # max_values = logits_tps[:, label].unsqueeze(1)
            # logits_tps = logits_tps / max_values
            avg_mav = torch.mean(logits_tps, dim=0)
            n_tps = logits_tps.shape[0]
            # features is running mean for mav
            self.features[label] = (
                self.features[label] * self.count[label] + avg_mav * n_tps
            )

            self.ex[label] += (logits_tps).sum(dim=0)
            self.ex2[label] += ((logits_tps) ** 2).sum(dim=0)
            self.count[label] += n_tps
            self.features[label] /= self.count[label] + 1e-8

    def forward(
        self, logits: torch.Tensor, sem_gt: torch.Tensor, is_train: torch.bool
    ) -> torch.Tensor:
        if is_train:
            # update mav only at training time
            sem_gt = sem_gt.type(torch.uint8)
            self.cumulate(logits, sem_gt)
        if self.previous_features == None:
            return torch.tensor(0.0).cuda()
        gt_labels = torch.unique(sem_gt).tolist()

        logits_permuted = logits.permute(0, 2, 3, 1)

        acc_loss = torch.tensor(0.0).cuda()
        for label in gt_labels[:-1]:
            mav = self.previous_features[label]
            logs = logits_permuted[torch.where(sem_gt == label)]
            mav = mav.expand(logs.shape[0], -1)
            if self.previous_count[label] > 0:
                ew_l1 = self.criterion(logs, mav)
                ew_l1 = ew_l1 / (self.var[label] + 1e-8)
                if self.hinged:
                    ew_l1 = F.relu(ew_l1 - self.delta).sum(dim=1)
                acc_loss += ew_l1.mean()

        return acc_loss

    def update(self):
        self.previous_features = self.features
        self.previous_count = self.count
        for c in self.var.keys():
            self.var[c] = (self.ex2[c] - self.ex[c] ** 2 / (self.count[c] + 1e-8)) / (
                self.count[c] + 1e-8
            )

        # resetting for next epoch
        self.count = torch.zeros(self.n_classes)  # count for class
        self.features = {
            i: torch.zeros(self.n_classes).cuda() for i in range(self.n_classes)
        }
        self.ex = {i: torch.zeros(self.n_classes).cuda() for i in range(self.n_classes)}
        self.ex2 = {
            i: torch.zeros(self.n_classes).cuda() for i in range(self.n_classes)
        }

        return self.previous_features, self.var

    def read(self):
        mav_tensor = torch.zeros(self.n_classes, self.n_classes)
        for key in self.previous_features.keys():
            mav_tensor[key] = self.previous_features[key]
        return mav_tensor

def calc_gamma(x, means, covs):
    """
    计算每个数据点属于每个高斯模型的可能性 γ_jk
    :param x: 数据集 (N, D) 维的张量，N 是数据点数量，D 是维度
    :param means: 高斯模型的均值 (K, D) 维的张量
    :param covs: 高斯模型的协方差矩阵 (K, D, D) 维的张量
    :param weights: 高斯模型的权重 α_k (K,) 维的张量
    :return: 归一化的 γ 值 (N, K) 维的张量
    """
    N, D = x.shape  # N 是数据点数量，D 是每个点的维度
    K = means.shape[0]  # K 是高斯模型的数量

    # 初始化 gamma 矩阵 (N, K) 用于存储每个数据点对每个模型的可能性
    gamma = torch.zeros(N, K)

    for k in range(K):
        # 对于每个高斯模型，创建一个多维正态分布
        mvn = dist.MultivariateNormal(means[k], covs[k])
        # 计算每个数据点属于该模型的可能性 φ(x_j | θ_k)
        pdf_values = mvn.log_prob(x).exp()
        # 乘以权重 α_k
        gamma[:, k] = pdf_values

    # 归一化以计算 γ_jk
    gamma_sum = gamma.sum(dim=1, keepdim=True)
    gamma_normalized = gamma / gamma_sum

    return gamma_normalized




# seg forward计算loss， gnn只统计
class MdsOWLoss(nn.Module):
    def __init__(self, unify_class, n_datasets, feature_dim, hinged=False, delta=0.1, ignore_index=255):
        super().__init__()
        self.unify_class = unify_class
        self.n_datasets = n_datasets
        self.feature_dim = feature_dim
        self.ignore_index = ignore_index

        self.hinged = hinged
        self.delta = delta
        self.count = torch.zeros(self.unify_class).cuda()  # count for class
        self.features = torch.zeros(self.unify_class, self.feature_dim).cuda()
        
        # See https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        # for implementation of Welford Alg.
        self.ex = torch.zeros(self.unify_class, self.feature_dim)
        self.ex2 = torch.zeros(self.unify_class, self.feature_dim)
        self.var = torch.zeros(self.unify_class, self.feature_dim)

        self.criterion = torch.nn.L1Loss(reduction="none")
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=ignore_index)

        self.previous_features = None
        self.previous_count = None

    def compute_loss(self, gamma, y_true, M):
        """
        并行计算损失函数 L_m
        :param gamma: (N, K) 的 γ 矩阵，每个数据点属于每个模型的可能性
        :param y_true: (N, C) 的真实标签，N 是数据点数，C 是类别数
        :param M: (K, C) 的模型输出矩阵
        :return: 标量的损失值
        """
        # 第一步：选择可能性最大的子模型 α
        alpha = torch.argmax(gamma, dim=1)  # (N,) 维的张量，代表每个数据点选择的模型
        
        keep_num = torch.sum(y_true!=self.ignore_index)

        # 第二步：选择可能性最大的模型输出, 并取 log_softmax
        # M[alpha[j]] 对应于每个数据点的子模型输出 (N, C)
        M_selected = M[alpha]  # 根据 alpha 选出每个数据点对应的模型输出 (N, C)
        
        loss = self.cross_entropy(M_selected, y_true)

        # 取 γ_jα 权重 (N,)
        gamma_jalpha = gamma.gather(1, alpha.unsqueeze(1)).squeeze(1).detach()  # (N,) 维

        # 计算最终的加权损失
        loss = torch.sum(gamma_jalpha * loss) / keep_num  # 标量

        return loss

    @torch.no_grad()
    def cumulate(self, logits: torch.Tensor, sem_gt: torch.Tensor):
        # Permute logits for easier access by sem_gt
        logits_permuted = logits.permute(0, 2, 3, 1)

        # Flatten logits and sem_gt for easier manipulation
        logits_flat = logits_permuted.reshape(-1, logits_permuted.shape[-1])  # Shape: (N, C)
        sem_gt_flat = sem_gt.view(-1)  # Shape: (N)

        # Get one-hot encoding of sem_gt_flat
        num_labels = logits.shape[1]
        sem_gt_one_hot = torch.nn.functional.one_hot(sem_gt_flat, num_classes=num_labels).float()  # Shape: (N, num_labels)

        # Compute the sum of logits and logits^2 for each label
        logits_sum = torch.einsum("nc,nl->lc", logits_flat, sem_gt_one_hot)  # Shape: (num_labels, C)
        logits_sq_sum = torch.einsum("nc,nl->lc", logits_flat**2, sem_gt_one_hot)  # Shape: (num_labels, C)

        # Count occurrences of each label
        label_count = sem_gt_one_hot.sum(dim=0)  # Shape: (num_labels)

        # Compute running mean for each label
        self.features = (self.features * self.count.unsqueeze(1) + logits_sum) / (self.count.unsqueeze(1) + label_count.unsqueeze(1) + 1e-8)

        # Update ex and ex2 for each label
        self.ex += logits_sum
        self.ex2 += logits_sq_sum

        # Update count for each label
        self.count += label_count
        

    def forward(
        self, unified_embedding: torch.Tensor, logits, gt, is_train, dataset_ids, seg_stage=True, adj_matrix=None
    ) -> torch.Tensor:
        acc_loss = torch.tensor(0.0).cuda()

        sem_gt = torch.argmax(logits, dim=1)

            
        if is_train:
            # update mav only at training time
            # this_sem_gt = unified_embedding.type(torch.uint8)
            self.cumulate(unified_embedding, sem_gt)
        

        if self.previous_features == None:
            return acc_loss
        
        if seg_stage:

            unified_embedding_permuted = unified_embedding.permute(0, 2, 3, 1)
            logits_flat = unified_embedding_permuted.reshape(-1, unified_embedding_permuted.shape[-1])
            sem_gt_flat = sem_gt.view(-1)
            
            indices = torch.nonzero(self.previous_count > 0).squeeze()
            
            select_sem_gt = sem_gt_flat[torch.isin(sem_gt_flat, indices)]

            num_labels = logits.shape[1]
            expand_mav = self.previous_features[select_sem_gt]

            expand_var = self.var[select_sem_gt]
            logs = logits_flat[torch.isin(sem_gt_flat, indices)]

            ew_l1 = self.criterion(logs, expand_mav)
            ew_l1 = ew_l1 / (expand_var + 1e-8)
            if self.hinged:
                ew_l1 = F.relu(ew_l1 - self.delta).sum(dim=1)
            acc_loss = ew_l1.mean()

        else:
            assert adj_matrix != None
            for i in range(self.n_datasets):
                if not (dataset_ids == i).any():
                    continue
                
                this_embedding = unified_embedding[dataset_ids==i]
                
                unified_embedding_permuted = this_embedding.permute(0, 2, 3, 1)
                unified_embedding_permuted = unified_embedding_permuted.reshape(-1, unified_embedding_permuted.shape[-1])
                covs = torch.diag(self.var)
                gamma_normalized = calc_gamma(unified_embedding_permuted, self.previous_features, covs)
                acc_loss = self.compute_loss(gamma_normalized, gt[dataset_ids==i], adj_matrix[i])
                

        return acc_loss

    def update(self):
        self.previous_features = self.features
        self.previous_count = self.count
        for i in range(self.n_datasets):
            for c in self.var[i].keys():
                self.var[i][c] = (self.ex2[i][c] - self.ex[i][c] ** 2 / (self.count[i][c] + 1e-8)) / (
                    self.count[i][c] + 1e-8
                )

        # resetting for next epoch
        self.count = torch.zeros(self.unify_class).cuda()  # count for class
        self.features = torch.zeros(self.unify_class, self.feature_dim).cuda()
        
        # See https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        # for implementation of Welford Alg.
        self.ex = torch.zeros(self.unify_class, self.feature_dim)
        self.ex2 = torch.zeros(self.unify_class, self.feature_dim)

        return self.previous_features, self.var

    def read(self):
        mav_tensor = torch.zeros(self.unify_class, self.unify_class)
        for key in self.previous_features.keys():
            mav_tensor[key] = self.previous_features[key]
        return mav_tensor

# seg forward计算loss， gnn只统计
class MdsOWLoss_cov(nn.Module):
    def __init__(self, unify_class, dataset_cats, feature_dim, hinged=False, delta=0.1):
        super().__init__()
        self.dataset_cats = dataset_cats
        self.unify_class = unify_class
        self.feature_dim = feature_dim
        self.n_datasets = len(dataset_cats)

        self.hinged = hinged
        self.delta = delta
        self.count = torch.zeros(self.unify_class)  # count for class
        self.features = torch.zeros(self.unify_class, self.feature_dim)
        self.var = torch.zeros(self.unify_class, self.feature_dim, self.feature_dim)

        self.criterion = torch.nn.L1Loss(reduction="none")

        self.previous_features = None
        self.previous_count = None
        self.previous_var = None

    @torch.no_grad()
    def cumulate(self, logits: torch.Tensor, sem_gt: torch.Tensor):
        # Permute logits for easier access by sem_gt
        logits_permuted = logits.permute(0, 2, 3, 1)

        # Flatten logits and sem_gt for easier manipulation
        logits_flat = logits_permuted.reshape(-1, logits_permuted.shape[-1])  # Shape: (N, C)
        sem_gt_flat = sem_gt.view(-1)  # Shape: (N)

        # Get one-hot encoding of sem_gt_flat
        num_labels = logits.shape[1]
        sem_gt_one_hot = torch.nn.functional.one_hot(sem_gt_flat, num_classes=num_labels).float()  # Shape: (N, num_labels)


        # Count occurrences of each label
        label_count = sem_gt_one_hot.sum(dim=0)  # Shape: (num_labels)
        means = (sem_gt_one_hot.T @ logits_flat) / (self.count.unsqueeze(1) + label_count.unsqueeze(1))
        
        
        # Compute running mean for each label
        self.features += means
        
        
        centered_data = logits_flat.unsqueeze(1) - self.features.unsqueeze(0)
        cov_updates = torch.einsum('nki,nkj->kij', centered_data, centered_data * sem_gt_one_hot.unsqueeze(-1))  # k x d x d
        self.var = self.var + cov_updates / (self.count.view(self.unify_class, 1, 1) + label_count.view(self.unify_class, 1, 1))


        # Update count for each label
        self.count += label_count
        

    def forward(
        self, unified_embedding: torch.Tensor, logits, gt, is_train, dataset_ids, seg_stage=True, adj_matrix=None
    ) -> torch.Tensor:
        acc_loss = torch.tensor(0.0).cuda()

        sem_gt = torch.argmax(logits, dim=1)

            
        if is_train:
            # update mav only at training time
            # this_sem_gt = unified_embedding.type(torch.uint8)
            self.cumulate(unified_embedding, sem_gt)
        

        if self.previous_features == None:
            return acc_loss
        
        if seg_stage:

            unified_embedding_permuted = unified_embedding.permute(0, 2, 3, 1)
            logits_flat = unified_embedding_permuted.reshape(-1, unified_embedding_permuted.shape[-1])
            sem_gt_flat = sem_gt.view(-1)
            
            indices = torch.nonzero(self.previous_count > 0).squeeze()
            
            select_sem_gt = sem_gt_flat[torch.isin(sem_gt_flat, indices)]

            num_labels = logits.shape[1]
            expand_mav = self.previous_features[select_sem_gt]

            expand_var = self.var[select_sem_gt]
            logs = logits_flat[torch.isin(sem_gt_flat, indices)]

            ew_l1 = self.criterion(logs, expand_mav)
            ew_l1 = ew_l1 / (expand_var + 1e-8)
            if self.hinged:
                ew_l1 = F.relu(ew_l1 - self.delta).sum(dim=1)
            acc_loss = ew_l1.mean()

        else:
            assert adj_matrix != None
            for i in range(self.n_datasets):
                if not (dataset_ids == i).any():
                    continue
                
                this_embedding = unified_embedding[dataset_ids==i]
                
                unified_embedding_permuted = this_embedding.permute(0, 2, 3, 1)
                unified_embedding_permuted = unified_embedding_permuted.reshape(-1, unified_embedding_permuted.shape[-1])
                gamma_normalized = calc_gamma(unified_embedding_permuted, self.previous_features, self.previous_var)
                acc_loss = self.compute_loss(gamma_normalized, gt[dataset_ids==i], adj_matrix[i])
                

        return acc_loss


    def update(self):
        self.previous_features = self.features
        self.previous_count = self.count
        self.previous_var = self.var


        self.count = torch.zeros(self.unify_class)  # count for class
        self.features = torch.zeros(self.unify_class, self.feature_dim)
        self.var = torch.zeros(self.unify_class, self.feature_dim, self.feature_dim)



        return self.previous_features, self.previous_var

    def read(self):
        mav_tensor = torch.zeros(self.unify_class, self.unify_class)
        for key in self.previous_features.keys():
            mav_tensor[key] = self.previous_features[key]
        return mav_tensor



