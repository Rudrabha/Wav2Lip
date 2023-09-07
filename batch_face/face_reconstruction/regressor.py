from batch_face.utils import is_box
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from . import mobilenet_v1
import torch.backends.cudnn as cudnn
import numpy as np
import cv2
from .utils import *


def preprocess(img, roi_box):
    crop = cv2.resize(
        crop_img(img, roi_box),
        dsize=(STD_SIZE, STD_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )
    return torch.from_numpy(crop.transpose((2, 0, 1))).float()


def postprocess(param, roi_box):
    param = param.detach().cpu().numpy()
    result = {}
    pts68 = predict_68pts(param, roi_box)[:2].T  # 68 pts
    result["pts68"] = pts68
    result["param"] = param
    result["roi_box"] = roi_box
    return result


class ShapeRegressor:
    def __init__(self, gpu_id=0, backend="3DDFA", file=None):
        if file is None:
            file = get_default_fr_file(backend)
        if backend == "3DDFA":
            arch = "mobilenet_1"
            checkpoint = torch.load(file, map_location=lambda storage, loc: storage)[
                "state_dict"
            ]
            model = getattr(mobilenet_v1, arch)(
                num_classes=62
            )  # 62 = 12(pose) + 40(shape) +10(expression)
            model_dict = {}
            # because the model is trained by multiple gpus, prefix module should be removed
            for k in checkpoint.keys():
                model_dict[k.replace("module.", "")] = checkpoint[k]
            model.load_state_dict(model_dict)
            if gpu_id >= 0:
                cudnn.benchmark = True
                model = model.cuda()
            model.eval()
            self.gpu_id = gpu_id
            self.model = model
        else:
            raise NotImplementedError(backend)

    def __call__(self, bbox, img):
        if is_box(bbox):
            bbox = [bbox]
            return self.batch_call(bbox, img)[0]
        else:
            return self.batch_call(bbox, img)

    def batch_call(self, boxes, img):
        roi_boxes = [parse_roi_box_from_bbox(bbox, img.shape[:2]) for bbox in boxes]
        input_imgs = [preprocess(img, roi_box) for roi_box in roi_boxes]
        input = torch.stack(input_imgs)
        if self.gpu_id >= 0:
            input = input.cuda(non_blocking=True)
        input = normalize(input)
        with torch.no_grad():
            params = self.model(input)
        assert len(params) == len(roi_boxes)
        results = [
            postprocess(param, roi_box) for param, roi_box in zip(params, roi_boxes)
        ]
        return results
