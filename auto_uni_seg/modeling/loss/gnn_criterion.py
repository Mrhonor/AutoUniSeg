import logging

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

from ..utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list



class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_ce": loss_ce}
        return losses
    
    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, unify_prototype, bi_graphs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        losses = {}
        if self.train_seg_or_gnn == self.GNN:
            if self.with_datasets_aux:
                logits = torch.einsum('bchw, nc -> bnhw', outputs['emb'], unify_prototype[self.total_cats:])
            else:
                logits = torch.einsum('bchw, nc -> bnhw', outputs['emb'], unify_prototype)
        else:
            remap_logits = outputs['logits']
            if self.with_datasets_aux:
                aux_logits_out = outputs['aux_logits']
            
        # remap_logits = []
        uot_rate = np.min([int(self.alter_iters) / self.first_stage_gnn_iters, 1])
        adj_rate = 1 - uot_rate
        cur_cat = 0
        for i in range(self.n_datasets):
            cur_cat += self.datasets_cats[i]
            
            if not (dataset_lbs == i).any():
                continue
            
            if self.train_seg_or_gnn == self.GNN:
                if len(bi_graphs) == 2*self.n_datasets:
                    remap_logits_1 = torch.einsum('bchw, nc -> bnhw', logits[dataset_lbs==i], bi_graphs[2*i])
                    remap_logits_2 = torch.einsum('bchw, nc -> bnhw', logits[dataset_lbs==i], bi_graphs[2*i+1])
                
                    remap_logits_1 = F.interpolate(remap_logits_1, size=(images.tensor.shape[2], images.tensor.shape[3]), mode="bilinear", align_corners=True)
                    loss_1 = self.criterion(remap_logits_1, targets[dataset_lbs==i])
                    
                    remap_logits_2 = F.interpolate(remap_logits_2, size=(images.tensor.shape[2], images.tensor.shape[3]), mode="bilinear", align_corners=True)
                    loss_2 = self.criterion(remap_logits_2, targets[dataset_lbs==i])
                    losses[f'loss_ce{i}'] = uot_rate*loss_1 + adj_rate*loss_2
                else:
                    remap_logits = torch.einsum('bchw, nc -> bnhw', logits[dataset_lbs==i], bi_graphs[i])
                
                    remap_logits = F.interpolate(remap_logits, size=(images.tensor.shape[2], images.tensor.shape[3]), mode="bilinear", align_corners=True)
                    loss = self.criterion(remap_logits, targets[dataset_lbs==i])
                    
                    losses[f'loss_ce{i}'] = loss
            else:
                remap_logits[i] = F.interpolate(remap_logits[i], size=(images.tensor.shape[2], images.tensor.shape[3]), mode="bilinear", align_corners=True)
                loss = self.criterion(remap_logits[i], targets[dataset_lbs==i])
                
                losses[f'loss_ce{i}'] = loss                        
    

            if self.with_datasets_aux:
                if self.train_seg_or_gnn == self.GNN:
                    aux_logits = torch.einsum('bchw, nc -> bnhw', outputs['emb'][dataset_lbs==i], unify_prototype[cur_cat-self.datasets_cats[i]:cur_cat])
                else:
                    aux_logits = aux_logits_out[i]
                
                aux_logits = F.interpolate(aux_logits, size=(images.tensor.shape[2], images.tensor.shape[3]), mode="bilinear", align_corners=True)
                aux_loss = self.criterion(aux_logits, targets[dataset_lbs==i])
                losses[f'loss_aux{i}'] = aux_loss
            

            
        
            if self.with_spa_loss and self.train_seg_or_gnn == self.GNN and self.inFirstGNNStage and self.iters > self.init_gnn_iters:
                
                if len(bi_graphs)==2*self.n_datasets:
                    spa_loss = torch.pow(torch.norm(bi_graphs[2*i+1], p='fro'), 2)
                else:
                    spa_loss =  torch.pow(torch.norm(bi_graphs[i], p='fro'), 2)
                
                losses['loss_spa'] = spa_loss
                
            if self.with_adj_loss and self.train_seg_or_gnn == self.GNN and self.inFirstGNNStage and self.iters > self.init_gnn_iters and self.target_bipart is not None:
                if len(bi_graphs) == 2*self.n_datasets:
                    total_num = bi_graphs[2*i+1].shape[0] * bi_graphs[2*i+1].shape[1]
                    base_weight = 1 / total_num
                    
                    if 'loss_adj' not in losses:
                        losses['loss_adj'] = base_weight * self.MSE_sum_loss(bi_graphs[2*i + 1][self.target_bipart[i] != 255], self.target_bipart[i][self.target_bipart[i] != 255])
                    else:
                        losses['loss_adj'] += base_weight * self.MSE_sum_loss(bi_graphs[2*i + 1][self.target_bipart[i] != 255], self.target_bipart[i][self.target_bipart[i] != 255])
                else:
                    total_num = bi_graphs[i].shape[0] * bi_graphs[i].shape[1]
                    base_weight = 1 / total_num
                    
                    if losses['loss_adj'] is None:
                        losses['loss_adj'] = base_weight * self.MSE_sum_loss(bi_graphs[2*i + 1][self.target_bipart[i] != 255], self.target_bipart[i][self.target_bipart[i] != 255])
                    else:
                        losses['loss_adj'] += base_weight * self.MSE_sum_loss(bi_graphs[2*i + 1][self.target_bipart[i] != 255], self.target_bipart[i][self.target_bipart[i] != 255])
        


        if self.with_orth_loss and self.train_seg_or_gnn == self.GNN:
            if self.with_datasets_aux:
                losses['loss_orth'] = self.similarity_dsb(unify_prototype[self.total_cats:])
            else:
                losses['loss_orth'] = self.similarity_dsb(unify_prototype)

        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)