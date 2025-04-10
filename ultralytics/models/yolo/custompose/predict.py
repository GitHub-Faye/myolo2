# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, LOGGER, ops


class CustomPosePredictor(DetectionPredictor):
    """
    A class extending the DetectionPredictor class for prediction based on a custom pose model.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.custompose import CustomPosePredictor

        args = dict(model="yolov12n-custompose.pt", source=ASSETS)
        predictor = CustomPosePredictor(overrides=args)
        predictor.predict_cli()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initializes CustomPosePredictor, sets task to 'custompose' and logs a warning for using 'mps' as device."""
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "custompose"
        
        # 获取并保存关键点形状，从多种可能的来源
        if hasattr(self.model, 'kpt_shape'):  # 直接属性访问（训练模式）
            self.kpt_shape = self.model.kpt_shape
        elif hasattr(self.model, 'yaml') and 'kpt_shape' in self.model.yaml:  # 从yaml配置获取
            self.kpt_shape = self.model.yaml['kpt_shape']
        else:  # 使用默认值
            self.kpt_shape = [5, 3]  # 设置为项目中实际使用的关键点数量
            LOGGER.warning(f"WARNING ⚠️ Could not determine kpt_shape, using default: {self.kpt_shape}")
        
        # MPS相关警告
        if isinstance(self.args.device, str) and self.args.device.lower() == "mps":
            LOGGER.warning(
                "WARNING ⚠️ Apple MPS known Pose bug. Recommend 'device=cpu' for Pose models. "
                "See https://github.com/ultralytics/ultralytics/issues/4031."
            )

    def postprocess(self, preds, img, orig_imgs):
        """Return detection results for a given input image or list of images."""
        preds = ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            classes=self.args.classes,
            nc=len(self.model.names),
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0]):
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape).round()
            # 使用self.kpt_shape而不是self.model.kpt_shape
            pred_kpts = pred[:, 6:].view(len(pred), *self.kpt_shape) if len(pred) else pred[:, 6:]
            pred_kpts = ops.scale_coords(img.shape[2:], pred_kpts, orig_img.shape)
            results.append(
                Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], keypoints=pred_kpts)
            )
        return results 