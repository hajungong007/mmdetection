
import cv2
import numpy as np
import logging
import os.path as osp
import tempfile

import mmcv
from torch.utils.data import Dataset

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from mmdet.core import eval_recalls

from mmdet.core import eval_map, eval_recalls
from .pipelines import Compose
from .registry import DATASETS
from .coco import CocoDataset
import random



import numba
from numba import jit, prange
import numpy as np
from collections import namedtuple
from typing import List, Union, Tuple


@jit(nopython=True)
def calculate_iou(gt, pr, form='pascal_voc') -> float:
    """Calculates the Intersection over Union.

    Args:
        gt: (np.ndarray[Union[int, float]]) coordinates of the ground-truth box
        pr: (np.ndarray[Union[int, float]]) coordinates of the prdected box
        form: (str) gt/pred coordinates format
            - pascal_voc: [xmin, ymin, xmax, ymax]
            - coco: [xmin, ymin, w, h]
    Returns:
        (float) Intersection over union (0.0 <= iou <= 1.0)
    """
    if form == 'coco':
        gt = gt.copy()
        pr = pr.copy()

        gt[2] = gt[0] + gt[2]
        gt[3] = gt[1] + gt[3]
        pr[2] = pr[0] + pr[2]
        pr[3] = pr[1] + pr[3]

    # Calculate overlap area
    dx = min(gt[2], pr[2]) - max(gt[0], pr[0]) + 1
    
    if dx < 0:
        return 0.0
    
    dy = min(gt[3], pr[3]) - max(gt[1], pr[1]) + 1

    if dy < 0:
        return 0.0

    overlap_area = dx * dy

    # Calculate union area
    union_area = (
            (gt[2] - gt[0] + 1) * (gt[3] - gt[1] + 1) +
            (pr[2] - pr[0] + 1) * (pr[3] - pr[1] + 1) -
            overlap_area
    )

    return overlap_area / union_area

@jit(nopython=True)
def find_best_match(gts, pred, pred_idx, threshold = 0.5, form = 'pascal_voc', ious=None) -> int:
    """Returns the index of the 'best match' between the
    ground-truth boxes and the prediction. The 'best match'
    is the highest IoU. (0.0 IoUs are ignored).

    Args:
        gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
        pred: (List[Union[int, float]]) Coordinates of the predicted box
        pred_idx: (int) Index of the current predicted box
        threshold: (float) Threshold
        form: (str) Format of the coordinates
        ious: (np.ndarray) len(gts) x len(preds) matrix for storing calculated ious.

    Return:
        (int) Index of the best match GT box (-1 if no match above threshold)
    """
    best_match_iou = -np.inf
    best_match_idx = -1

    for gt_idx in range(len(gts)):
        
        if gts[gt_idx][0] < 0:
            # Already matched GT-box
            continue
        
        iou = -1 if ious is None else ious[gt_idx][pred_idx]

        if iou < 0:
            iou = calculate_iou(gts[gt_idx], pred, form=form)
            
            if ious is not None:
                ious[gt_idx][pred_idx] = iou

        if iou < threshold:
            continue

        if iou > best_match_iou:
            best_match_iou = iou
            best_match_idx = gt_idx

    return best_match_idx

@jit(nopython=True)
def calculate_precision(gts, preds, threshold = 0.5, form = 'coco', ious=None) -> float:
    """Calculates precision for GT - prediction pairs at one threshold.

    Args:
        gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
        preds: (List[List[Union[int, float]]]) Coordinates of the predicted boxes,
               sorted by confidence value (descending)
        threshold: (float) Threshold
        form: (str) Format of the coordinates
        ious: (np.ndarray) len(gts) x len(preds) matrix for storing calculated ious.

    Return:
        (float) Precision
    """
    n = len(preds)
    tp = 0
    fp = 0
    
    # for pred_idx, pred in enumerate(preds_sorted):
    for pred_idx in range(n):

        best_match_gt_idx = find_best_match(gts, preds[pred_idx], pred_idx,
                                            threshold=threshold, form=form, ious=ious)

        if best_match_gt_idx >= 0:
            # True positive: The predicted box matches a gt box with an IoU above the threshold.
            tp += 1
            # Remove the matched GT box
            gts[best_match_gt_idx] = -1

        else:
            # No match
            # False positive: indicates a predicted box had no associated gt box.
            fp += 1

    # False negative: indicates a gt box had no associated predicted box.
    fn = (gts.sum(axis=1) > 0).sum()

    return tp / (tp + fp + fn)


@jit(nopython=True)
def calculate_image_precision(gts, preds, thresholds = (0.5, ), form = 'coco') -> float:
    """Calculates image precision.

    Args:
        gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
        preds: (List[List[Union[int, float]]]) Coordinates of the predicted boxes,
               sorted by confidence value (descending)
        thresholds: (float) Different thresholds
        form: (str) Format of the coordinates

    Return:
        (float) Precision
    """
    n_threshold = len(thresholds)
    image_precision = 0.0
    
    ious = np.ones((len(gts), len(preds))) * -1
    # ious = None

    for threshold in thresholds:
        precision_at_threshold = calculate_precision(gts.copy(), preds, threshold=threshold,
                                                     form=form, ious=ious)
        image_precision += precision_at_threshold / n_threshold

    return image_precision


    
    
@DATASETS.register_module
class WheatDataset(CocoDataset):
    """Car dataset."""
    CLASSES = ('whead')
    def __init__(self,
                 ann_file,
                 pipeline,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True,
                 mosaic=False):
        super().__init__(ann_file,
                 pipeline,
                 data_root,
                 img_prefix,
                 seg_prefix,
                 proposal_file,
                 test_mode,
                 filter_empty_gt)
    
        self.mosaic = mosaic
        if self.mosaic:
            self.load_pipeline = Compose(pipeline[:2])
            self.transform_pipeline = Compose(pipeline[2:])
            self.mosaic_size = 1024
        
    
    def load_mosaic(self, index):
        # loads images in a mosaic
        
        bboxes4 = []
        labels4 = []
        s = self.mosaic_size
        xc, yc = [int(random.uniform(s * 0.5, s * 1.5)) for _ in range(2)]  # mosaic center x, y
        indices = [index] + [random.randint(0, len(self.img_infos) - 1) for _ in range(3)]  # 3 additional image indices
        for i, index in enumerate(indices):
            # Load image
            img_info = self.img_infos[index]
            ann_info = self.get_ann_info(index)
            results = dict(img_info=img_info, ann_info=ann_info)
            self.pre_pipeline(results)
            results = self.load_pipeline(results)
            
            img = results['img']
            h, w = img.shape[:2]
            
            # place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
    
            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b
    
            # Labels
            gt_bboxes = results['gt_bboxes']
            gt_labels = results['gt_labels']
            bboxes = gt_bboxes.copy()
            if bboxes.size > 0:  # Normalized xywh to pixel xyxy format
                bboxes[:, ::2] += padw
                bboxes[:, 1::2] += padh
            labels4.append(gt_labels)
            bboxes4.append(bboxes)
    
        # Augment
        start = s//2
        end = int(s * 1.5)
        img4 = img4[start: end, start:end]  # center crop (WARNING, requires box pruning)
        
        # TODO: 防止无bboxes
        if len(labels4):
            labels4 = np.concatenate(labels4, 0)
            bboxes4 = np.concatenate(bboxes4, 0)
            # np.clip(labels4[:, 1:] - s / 2, 0, s, out=labels4[:, 1:])  # use with center crop
            np.clip(bboxes4[:, 0:], start, end, out=bboxes4[:, 0:])  # use with random_affine
            
        # Concat/clip labels
        mosaic_labels = []
        mosaic_bboxes = []
        for label, bbox in zip(labels4, bboxes4):
            x1,y1,x2,y2 = bbox
            if y2-y1==0 or x2-x1==0:
                continue
            if x1 > end or x2 < start or y1 > end or y2 < start:
                continue
            mosaic_bboxes.append(bbox)
            mosaic_labels.append(label)
        mosaic_bboxes = np.array(mosaic_bboxes) - start
        mosaic_labels = np.array(mosaic_labels)
        
        if len(mosaic_bboxes)==0:
            mosaic_bboxes = np.zeros((0, 4), dtype=np.float32)
            mosaic_labels = np.array([], dtype=np.int64)
            
        mosaic_result=dict()
        mosaic_result['img'] = img4
        mosaic_result['img_shape'] = img4.shape
        mosaic_result['ori_shape'] = img4.shape
        mosaic_result['pad_shape'] = img4.shape
        mosaic_result['scale_factor'] = 1.0
        mosaic_result['img_norm_cfg'] = results['img_norm_cfg']
        mosaic_result['gt_bboxes'] = mosaic_bboxes
        mosaic_result['gt_labels'] = mosaic_labels
        mosaic_result['filename'] = ''
        mosaic_result['bbox_fields'] = []
        mosaic_result['bbox_fields'].append('gt_bboxes')
        #img4, labels4 = random_affine(img4, labels4,
        #                              degrees=1.98 * 2,
        #                              translate=0.05 * 2,
        #                              scale=0.05 * 2,
        #                              shear=0.641 * 2,
        #                              border=-s // 2)  # border to remove
    
        return mosaic_result
    
    def prepare_train_img(self, idx):
                   
        if self.mosaic and random.randint(0,1) ==0:
        #if self.mosaic:
            results = self.load_mosaic(idx)
            results = self.transform_pipeline(results)
            return results
            
        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        
        self.pre_pipeline(results)
        return self.pipeline(results)
    
    
    


    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=np.arange(0.5, 0.96, 0.05)):
        """Evaluation in COCO protocol.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float]): IoU threshold used for evaluating
                recalls. If set to a list, the average recall of all IoUs will
                also be computed. Default: 0.5.

        Returns:
            dict[str: float]
        """
        super_eval_results = super().evaluate(results,
                                         metric='bbox',
                                         logger=None,
                                         jsonfile_prefix=None,
                                         classwise=False,
                                         proposal_nums=(100, 300, 1000),
                                         iou_thrs=np.arange(0.5, 0.96, 0.05))
        
        iou_thresholds = numba.typed.List()
        for x in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75]:
            iou_thresholds.append(x)
        
        eval_results = dict()
        # load gts once
        gts = []
        for i in range(len(results) ):
            # load gts
            bboxes = self.coco.loadAnns(self.coco.getAnnIds(i))
            # bboxes format:
            # [{'id': 0, 'image_id': 0,'category_id': 1,'segmentation': [[294.0, 61.0, 381.0, 61.0, 381.0, 181.0, 294.0, 181.0]],
            #  'bbox': [294.0, 61.0, 88.0, 121.0],'iscrowd': 0,'area': 10648.0}]
            
            # xywh
            bboxes = [ bbox['bbox']  for bbox in bboxes]
            bboxes = np.array(bboxes)
            
            # xywh ==> x1y1x2y2
            bboxes[:,2] = bboxes[:,0] + bboxes[:,2]
            bboxes[:,3] = bboxes[:,1] + bboxes[:,3]
            gts.append(bboxes)
        
        score_thresholds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
        max_precisions, best_thres = 0, 0
        for i, score_thres in enumerate(score_thresholds):
            precisions = []
            for gt, result in zip(gts, results):
                # load preds
                pred = np.array(result[0])
                pred = pred[pred[:,4]>score_thres]
                pred = pred[:,:4]
            
                precision = calculate_image_precision(gt, pred, thresholds=iou_thresholds, form='pascal_voc')
                precisions.append(precision)
       
            precisions = np.mean(precisions)
        
            if precisions > max_precisions:
                max_precisions = precisions
                best_thres = score_thres
            
            item = 'score_thres@{:.2f}'.format(score_thres)
            eval_results[item] = '{:.4f}'.format(precisions)
            
        eval_results['best_thres']= '@{:.1f}, {:.4f}'.format(best_thres, max_precisions)
        print(eval_results)
        
        #eval_results.update(super_eval_results)
            
        return eval_results
