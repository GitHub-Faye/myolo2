# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from pathlib import Path

import numpy as np
import torch

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import OKS_SIGMA, PoseMetrics, box_iou, kpt_iou
from ultralytics.utils.plotting import output_to_target, plot_images


class CustomPoseValidator(DetectionValidator):
    """
    A class extending the DetectionValidator class for validation based on a custom pose model.

    Example:
        ```python
        from ultralytics.models.yolo.custompose import CustomPoseValidator

        args = dict(model="yolov12n-custompose.pt", data="coco8-pose.yaml")
        validator = CustomPoseValidator(args=args)
        validator()
        ```
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize a 'CustomPoseValidator' object with custom parameters and assigned attributes."""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.sigma = None
        self.kpt_shape = None
        self.args.task = "custompose"
        self.metrics = PoseMetrics(save_dir=self.save_dir, on_plot=self.on_plot)
        if isinstance(self.args.device, str) and self.args.device.lower() == "mps":
            LOGGER.warning(
                "WARNING ⚠️ Apple MPS known Pose bug. Recommend 'device=cpu' for Pose models. "
                "See https://github.com/ultralytics/ultralytics/issues/4031."
            )

    def preprocess(self, batch):
        """Preprocesses the batch by converting the 'keypoints' data into a float and moving it to the device."""
        batch = super().preprocess(batch)
        batch["keypoints"] = batch["keypoints"].to(self.device).float()
        return batch

    def get_desc(self):
        """Returns description of evaluation metrics in string format."""
        return ("%22s" + "%11s" * 10) % (
            "Class",
            "Images",
            "Instances",
            "Box(P",
            "R",
            "mAP50",
            "mAP50-95)",
            "Pose(P",
            "R",
            "mAP50",
            "mAP50-95)",
        )

    def postprocess(self, preds):
        """Apply non-maximum suppression and return detections with high confidence scores."""
        return ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            labels=self.lb,
            multi_label=True,
            agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=self.nc,
        )

    def init_metrics(self, model):
        """Initiate pose estimation metrics for YOLO model."""
        super().init_metrics(model)
        self.kpt_shape = self.data["kpt_shape"]
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]
        self.sigma = OKS_SIGMA if is_pose else np.ones(nkpt) / nkpt
        self.stats = dict(tp_p=[], tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])

    def _prepare_batch(self, si, batch):
        """Prepares a batch for processing by converting keypoints to float and moving to device."""
        pbatch = super()._prepare_batch(si, batch)
        kpts = batch["keypoints"][batch["batch_idx"] == si]
        h, w = pbatch["imgsz"]
        kpts = kpts.clone()
        kpts[..., 0] *= w
        kpts[..., 1] *= h
        kpts = ops.scale_coords(pbatch["imgsz"], kpts, pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"])
        pbatch["kpts"] = kpts
        return pbatch

    def _prepare_pred(self, pred, pbatch):
        """Prepares and scales keypoints in a batch for pose processing."""
        predn = super()._prepare_pred(pred, pbatch)
        nk = pbatch["kpts"].shape[1]
        pred_kpts = predn[:, 6:].view(len(predn), nk, -1)
        ops.scale_coords(pbatch["imgsz"], pred_kpts, pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"])
        return predn, pred_kpts

    def update_metrics(self, preds, batch):
        """Metrics."""
        for si, pred in enumerate(preds):
            self.seen += 1
            npr = len(pred)
            stat = dict(
                conf=torch.zeros(0, device=self.device),
                pred_cls=torch.zeros(0, device=self.device),
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
                tp_p=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
            )
            pbatch = self._prepare_batch(si, batch)
            cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
            nl = len(cls)
            stat["target_cls"] = cls
            stat["target_img"] = cls.unique()
            if npr == 0:
                if nl:
                    for k in self.stats.keys():
                        self.stats[k].append(stat[k])
                    if self.args.plots:
                        self.confusion_matrix.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls)
                continue

            # Predictions
            if self.args.single_cls:
                pred[:, 5] = 0
            predn, pred_kpts = self._prepare_pred(pred, pbatch)
            stat["conf"] = predn[:, 4]
            stat["pred_cls"] = predn[:, 5]

            # Evaluate
            if nl:
                stat["tp"] = self._process_batch(predn, bbox, cls)
                stat["tp_p"] = self._process_batch(predn, bbox, cls, pred_kpts, pbatch["kpts"])
            if self.args.plots:
                self.confusion_matrix.process_batch(predn, bbox, cls)

            for k in self.stats.keys():
                self.stats[k].append(stat[k])

            # Save
            if self.args.save_json:
                self.pred_to_json(predn, batch["im_file"][si])
            if self.args.save_txt:
                self.save_one_txt(
                    predn,
                    pred_kpts,
                    self.args.save_conf,
                    pbatch["ori_shape"],
                    self.save_dir / "labels" / f"{Path(batch['im_file'][si]).stem}.txt",
                )

    def _process_batch(self, detections, gt_bboxes, gt_cls, pred_kpts=None, gt_kpts=None):
        """
        Return correct prediction matrix by computing Intersection over Union (IoU) between detections and ground truth.

        Args:
            detections (torch.Tensor): Tensor with shape (N, 6) representing detection boxes and scores, where each
                detection is of the format (x1, y1, x2, y2, conf, class).
            gt_bboxes (torch.Tensor): Tensor with shape (M, 4) representing ground truth bounding boxes, where each
                box is of the format (x1, y1, x2, y2).
            gt_cls (torch.Tensor): Tensor with shape (M,) representing ground truth class indices.
            pred_kpts (torch.Tensor | None): Optional tensor with shape (N, 51) representing predicted keypoints, where
                51 corresponds to 17 keypoints each having 3 values.
            gt_kpts (torch.Tensor | None): Optional tensor with shape (N, 51) representing ground truth keypoints.

        Returns:
            torch.Tensor: A tensor with shape (N, 10) representing the correct prediction matrix for 10 IoU levels,
                where N is the number of detections.
        """
        if pred_kpts is not None and gt_kpts is not None:
            # `0.53` is from https://github.com/jin-s13/xtcocoapi/blob/master/xtcocotools/cocoeval.py#L384
            area = ops.xyxy2xywh(gt_bboxes)[:, 2:].prod(1) * 0.53
            iou = kpt_iou(gt_kpts, pred_kpts, sigma=self.sigma, area=area)
        else:  # boxes
            iou = box_iou(gt_bboxes, detections[:, :4])

        return self.match_predictions(detections[:, 5], gt_cls, iou)

    def plot_val_samples(self, batch, ni):
        """Plots and saves validation set samples with predicted bounding boxes and keypoints."""
        plot_images(
            batch["img"],
            batch["batch_idx"],
            batch["cls"].squeeze(-1),
            batch["bboxes"],
            kpts=batch["keypoints"],
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def plot_predictions(self, batch, preds, ni):
        """Plots predictions for YOLO model."""
        pred_kpts = torch.cat([p[:, 6:].view(-1, *self.kpt_shape) for p in preds], 0)
        plot_images(
            batch["img"],
            *output_to_target(preds, max_det=self.args.max_det),
            kpts=pred_kpts,
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )  # pred

    def save_one_txt(self, predn, pred_kpts, save_conf, shape, file):
        """Save YOLO detections to a txt file in normalized coordinates as (cls, box_and_conf, num_keypoints * kpts)."""
        gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
        for i, (pred, pred_kpt) in enumerate(zip(predn, pred_kpts)):
            coords = []
            line = (pred[5], *(pred[:4] / gn))
            line = (i, pred[5], *(pred[:4] / gn))
            for kpt in pred_kpt:
                coords.extend(kpt[:2] / gn[[0, 1]])
                coords.append(kpt[2])
            line = [line[1], *line[3:], 0] if len(coords) == 0 else [line[1], *line[3:], *coords]  # convert to xywh
            if save_conf:
                line.insert(2, pred[4])  # confidence
            with open(file, "a") as f:
                f.write(("%g " * len(line)).rstrip() % tuple(line) + "\n")

    def pred_to_json(self, predn, filename):
        """Convert predictions to JSON for COCO evaluation."""
        stem = Path(filename).stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = ops.xyxy2xywh(predn[:, :4])  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        for p, b in zip(predn.tolist(), box.tolist()):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "category_id": int(p[5]),
                    "bbox": [round(x, 3) for x in b],
                    "score": round(p[4], 5),
                }
            )

    def eval_json(self, stats):
        """Evaluates object detection model using COCO evaluation."""
        if self.args.save_json and self.is_coco and len(self.jdict):
            anno_json = self.data["path"] / "annotations/instances_val2017.json"  # annotations
            pred_json = self.save_dir / "predictions.json"  # predictions
            LOGGER.info(f"\nEvaluating pycocotools mAP using {pred_json} and {anno_json}...")
            try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
                check_requirements("pycocotools>=2.0.6")
                from pycocotools.coco import COCO  # noqa
                from pycocotools.cocoeval import COCOeval  # noqa

                for x in pred_json, anno_json:
                    assert x.is_file(), f"{x} file not found"
                with open(pred_json, "w") as f:
                    json.dump(self.jdict, f)
                stats = []  # waiting for datasets to be ready before running eval
            except Exception as e:
                LOGGER.warning(f"WARNING ⚠️ pycocotools unable to run: {e}")
        return stats 