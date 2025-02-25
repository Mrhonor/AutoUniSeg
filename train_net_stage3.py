try:
    # ignore ShapelyDeprecationWarning from fvcore
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except:
    pass

import copy
import itertools
import logging
import os

from collections import OrderedDict
from typing import Any, Dict, List, Set

import torch
import torch.nn as nn

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import (
    DatasetEvaluators,
    SemSegEvaluator,
    verify_results,
    
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger



from auto_uni_seg import (
    COCOInstanceNewBaselineDatasetMapper,
    COCOPanopticNewBaselineDatasetMapper,
    InstanceSegEvaluator,
    MaskFormerInstanceDatasetMapper,
    MaskFormerPanopticDatasetMapper,
    MaskFormerSemanticDatasetMapper,
    MaskFormerSemanticDatasetMapper_2,
    SemanticDatasetMapper,
    add_maskformer2_config,
    add_hrnet_config,
    add_gnn_config,
    add_afformer_config,
    add_segmenter_comfig,
    LoaderAdapter,
    build_bipartite_graph_for_unseen,
    eval_for_mseg_datasets,
    UniDetLearnUnifyLabelSpace
)


from PIL import Image
from detectron2.utils.file_io import PathManager
import numpy as np
from functools import partial
from detectron2.structures import ImageList
import torch.nn.functional as F
import logging

from auto_uni_seg.utils.evaluate import eval_link_hook, iter_info_hook, find_unuse_hook, print_unify_label_space, print_bipartite
from contextlib import ExitStack, contextmanager
import time
import datetime
from detectron2.utils.logger import log_every_n_seconds

logger = logging.getLogger(__name__)
def my_sem_seg_loading_fn(filename, dtype=int, lb_map=None, size_divisibility=-1, ignore_label=255):
    with PathManager.open(filename, "rb") as f:
        image = np.array(Image.open(f), copy=False, dtype=dtype)
        if lb_map is not None:
            image = lb_map[image] 

    #     logger.info(f'size_divisibility: {size_divisibility}')
    #     if size_divisibility > 0:
    #         image = torch.tensor(image)
            
    #         image_size = (image.shape[0], image.shape[1])
    #         padding_size = [
    #             0,
    #             size_divisibility - image_size[1],
    #             0,
    #             size_divisibility - image_size[0],
    #         ]
            
    #         image = F.pad(image, padding_size, value=ignore_label).contiguous()
    #         logger.info(f'image shape: {image.shape}')
    #         image = image.numpy()

    # dsaf
    return image
    
@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)

class Trainer(DefaultTrainer):


    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each
        builtin dataset. For your own dataset, you can simply create an
        evaluator manually in your script and do not have to worry about the
        hacky if-else logic here.
        """
        logger.info(f"build evaluator:{dataset_name}")
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        # semantic segmentation
        if evaluator_type in ["sem_seg", ]:
            lb_map = np.arange(256).astype(np.uint8)
            lookup_table = MetadataCatalog.get(dataset_name).thing_dataset_id_to_contiguous_id
            for k, v in lookup_table.items():
                lb_map[k] = v
            logger.info(f"evaluator_type:{dataset_name}")
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                    sem_seg_loading_fn=partial(my_sem_seg_loading_fn, lb_map=lb_map, size_divisibility=cfg.INPUT.SIZE_DIVISIBILITY, ignore_label=cfg.DATASETS.IGNORE_LB)
                )
            )
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        # Semantic segmentation dataset mapper
        if cfg.INPUT.DATASET_MAPPER_NAME == 'BASE':
            return LoaderAdapter(cfg, aux_mode='train')
        else:
            mapper = None
            return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        if 'cs' in dataset_name:
            dataset_id = 0            
        elif 'mapi' in dataset_name:
            dataset_id = 1
        elif 'sunrgbd' in dataset_name:
            dataset_id = 2
        elif 'bdd' in dataset_name:
            dataset_id = 3
        elif 'idd' in dataset_name:
            dataset_id = 4
            # dataset_id = 1
        elif 'ade' in dataset_name:
            dataset_id = 5
        elif 'coco' in dataset_name:
            dataset_id = 6
        else:
            dataset_id = 0
        # dataset_id = 0
        aux_mode = 'test'
        if '_2' in dataset_name:
            aux_mode = 'eval'
            
        # return LoaderAdapter(cfg, aux_mode=aux_mode, dataset_id=dataset_id, datasets_name=[dataset_name])
        return LoaderAdapter(cfg, aux_mode=aux_mode, dataset_id=dataset_id)

    @classmethod
    def build_eval_loader(cls, cfg, dataset_name):
        if 'cs' in dataset_name:
            dataset_id = 0            
        elif 'mapi' in dataset_name:
            dataset_id = 1
        elif 'sunrgbd' in dataset_name:
            dataset_id = 2
        elif 'bdd' in dataset_name:
            dataset_id = 3
        elif 'idd' in dataset_name:
            dataset_id = 4
        elif 'ade' in dataset_name:
            dataset_id = 5
        elif 'coco' in dataset_name:
            dataset_id = 6
        else:
            dataset_id = 0

        aux_mode = 'eval'
            
        return LoaderAdapter(cfg, aux_mode=aux_mode, dataset_id=dataset_id)

    @classmethod
    def find_unuse_link(self, cfg, model):
        # if torch.distributed.is_initialized():
        #     model = self.model.module
        # else:
        #     model = self.model

        logger = logging.getLogger(__name__)

        # model.finetune_stage = torch.zeros(1)
        bipart_graph = model.get_bipart_graph()
        callbacks = None
        ignore_label = 255
        datasets_cats = cfg.DATASETS.DATASETS_CATS
        n_datasets = len(datasets_cats)
        ignore_index = cfg.DATASETS.IGNORE_LB
        total_cats = 0
        for i in range(0, n_datasets):
            total_cats += datasets_cats[i]
        num_unfiy_class = cfg.DATASETS.NUM_UNIFY_CLASS
        datasets_name = cfg.DATASETS.TRAIN
        # dls = get_data_loader(configer, aux_mode='train', distributed=is_dist, stage=2)
        print_bipartite(datasets_cats, n_datasets, bipart_graph, total_cats, datasets_name)
        # return

        loaded_map = {}
        for dataset_idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            # if dataset_idx < 5:
            #     continue
            logger.info("evaluating dataset {}:".format(i+1))    

            data_loader = self.build_test_loader(cfg, dataset_name)
    
            n_classes = datasets_cats[dataset_idx]
            hist = torch.zeros(n_classes, num_unfiy_class).cuda()    
            
            with torch.no_grad():
                total = len(data_loader)
                num_warmup = min(5, total - 1)
                start_time = time.perf_counter()
                total_data_time = 0
                total_compute_time = 0
                total_eval_time = 0
                with ExitStack() as stack:
                    if isinstance(model, nn.Module):
                        stack.enter_context(inference_context(model))
                    stack.enter_context(torch.no_grad())

                    start_data_time = time.perf_counter()
                    dict.get(callbacks or {}, "on_start", lambda: None)()
                    for idx, inputs in enumerate(data_loader):
                        total_data_time += time.perf_counter() - start_data_time
                        if idx == num_warmup:
                            start_time = time.perf_counter()
                            total_data_time = 0
                            total_compute_time = 0
                            total_eval_time = 0
                        for x in inputs:
                            im = x["image"]
                            if im.shape[-2] > 2200 or im.shape[-1] > 2200:
                                x["image"] = F.interpolate(im[None], size=(int(im.shape[-2]*0.5), int(im.shape[-1]*0.5)), mode='bilinear', align_corners=True).squeeze(0)
                                x["sem_seg"] = F.interpolate(x["sem_seg"].float()[None][None], size=(int(im.shape[-2]*0.5), int(im.shape[-1]*0.5)), mode='nearest').squeeze().long()
                                x["height"] = int(x["height"]*0.5)
                                x["width"] = int(x["width"]*0.5)
                                
                        start_compute_time = time.perf_counter()
                        dict.get(callbacks or {}, "before_inference", lambda: None)()
                        outputs = model(inputs)
                        dict.get(callbacks or {}, "after_inference", lambda: None)()
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        total_compute_time += time.perf_counter() - start_compute_time

                        start_eval_time = time.perf_counter()
                        labels = [x["sem_seg"][None].cuda() for x in inputs]

                        logits = [output["uni_logits"][None] for output in outputs]
                        
                        for lb, lg in zip(labels, logits):
                            # print(lb.shape)
                            # print(lg.shape)
                            lb = F.interpolate(lb.unsqueeze(1).float(), size=(lg.shape[2], lg.shape[3]),
                                    mode='nearest').squeeze(1).long()

                            probs = torch.softmax(lg, dim=1)
                            preds = torch.argmax(probs, dim=1)
                                                
                            keep = lb != ignore_label

                            hist += torch.tensor(np.bincount(
                                lb.cpu().numpy()[keep.cpu().numpy()] * num_unfiy_class + preds.cpu().numpy()[keep.cpu().numpy()],
                                minlength=n_classes * num_unfiy_class
                            )).cuda().view(n_classes, num_unfiy_class) 
                        total_eval_time += time.perf_counter() - start_eval_time

                        iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
                        data_seconds_per_iter = total_data_time / iters_after_start
                        compute_seconds_per_iter = total_compute_time / iters_after_start
                        eval_seconds_per_iter = total_eval_time / iters_after_start
                        total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
                        if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                            eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                            log_every_n_seconds(
                                logging.INFO,
                                (
                                    f"Inference done {idx + 1}/{total}. "
                                    f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                                    f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                                    f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                                    f"Total: {total_seconds_per_iter:.4f} s/iter. "
                                    f"ETA={eta}"
                                ),
                                n=5,
                            )
                        start_data_time = time.perf_counter()
                    dict.get(callbacks or {}, "on_end", lambda: None)()

                # Measure the time only for this worker (before the synchronization barrier)
                total_time = time.perf_counter() - start_time
                total_time_str = str(datetime.timedelta(seconds=total_time))
                # NOTE this format is parsed by grep
                logger.info(
                    "Total inference time: {} ({:.6f} s / iter per device)".format(
                        total_time_str, total_time / (total - num_warmup)
                    )
                )
                total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
                logger.info(
                    "Total inference pure compute time: {} ({:.6f} s / iter per device)".format(
                        total_compute_time_str, total_compute_time / (total - num_warmup)
                    )
                )
            
            max_value, max_index = torch.max(bipart_graph[dataset_idx], dim=0)
            # print(max_value)
            
            # torch.set_printoptions(profile="full")
            # print(hist)

            buckets = {}
            for index, j in enumerate(max_index):
                if max_value[index] == 0:
                    continue
                
                if int(j) not in buckets:
                    buckets[int(j)] = [index]
                else:
                    buckets[int(j)].append(index)

            for index in range(0, n_classes):
                if index not in buckets:
                    logger.info(f'index not in buckets: {index}')
                    buckets[index] = []

            for index, val in buckets.items():
                total_num = 0
                for i in val:
                    total_num += hist[index][i]
                new_val = []
                if total_num != 0:
                    for i in val:
                        rate = hist[index][i] / total_num
                        if rate > 0.0001:
                            # new_val.append([i, rate])
                            new_val.append(i)
                # else:
                #     for i in val:
                #         # new_val.append([i, 0])
                #         new_val.append(i)
                
                buckets[index] = new_val
                
                
            for index in range(0, n_classes):
                if index not in buckets:
                    buckets[index] = []
                print("\"{}\": {}".format(index, buckets[index]))    
            
            loaded_map[f'dataset{dataset_idx}'] = buckets

        bi_graphs = []
        for dataset_id in range(0, n_datasets):
            n_cats = datasets_cats[dataset_id]
            this_bi_graph = torch.zeros(n_cats, num_unfiy_class)
            for key, val in loaded_map['dataset'+str(dataset_id)].items():
                this_bi_graph[int(key)][val] = 1
                
            bi_graphs.append(this_bi_graph.cuda())

        model.set_bipartite_graphs(bi_graphs) 
        torch.save(model.state_dict(), 'output/find_unuse_7ds_model_hrnet.pth')


    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad and 'adj_matrix' not in module_param_name:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer



def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_hrnet_config(cfg)
    add_afformer_config(cfg)
    add_maskformer2_config(cfg)
    add_gnn_config(cfg)
    add_segmenter_comfig(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="auto_uni_seg")
    return cfg

def build_bipart_for_unseen(cfg, model):
    """
    Build bipartite graph for unseen classes.
    """
    from auto_uni_seg.utils import build_bipartite_graph_for_unseen
    build_bipartite_graph_for_unseen(cfg, model)
    

def main(args):
    cfg = setup(args)
    
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        # eval_for_mseg_datasets(Trainer.build_test_loader, cfg, model)
        # Trainer.find_unuse_link(cfg, model)
        # if args.unseen:
        #     build_bipartite_graph_for_unseen(Trainer.build_test_loader, cfg, model)
        # print_unify_label_space(Trainer.build_test_loader, model, cfg)
        # return
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res
        # return
    
    trainer = Trainer(cfg)
    trainer.register_hooks([find_unuse_hook(), iter_info_hook()])
    # trainer.register_hooks([iter_info_hook()])
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()

def argument_parser():
    parser = default_argument_parser()
    parser.add_argument(
        "--unseen",
        action="store_true",
        help="Whether to evaluate unseen datasets. ",
    )
    return parser


if __name__ == "__main__":
    args = argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
