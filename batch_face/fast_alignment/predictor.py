import torch
import cv2
import numpy as np
from torch.utils.data import DataLoader
from .basenet import MobileNet_GDConv
from .pfld_compressed import PFLDInference
from batch_face.utils import (
    detection_adapter,
    get_default_onnx_file,
    is_image,
    is_box,
    to_numpy,
    load_weights,
)

try:
    import onnx
    import onnxruntime
except:
    onnx = None
    onnxruntime = None


def get_device(gpu_id):
    if gpu_id > -1:
        return torch.device("cuda:%s" % str(gpu_id))
    else:
        return torch.device("cpu")


# landmark of (5L, 2L) from [0,1] to real range
def reproject(bbox, landmark):
    landmark_ = (
        landmark.clone() if isinstance(landmark, torch.Tensor) else landmark.copy()
    )
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    landmark_[:, 0] *= float(w)
    landmark_[:, 0] += float(x1)
    landmark_[:, 1] *= float(h)
    landmark_[:, 1] += float(y1)
    return landmark_


def prepare_feed(img, face, backbone):
    if backbone == "MobileNet":
        out_size = 224
    else:
        out_size = 112
    mean = np.asarray([0.485, 0.456, 0.406])
    std = np.asarray([0.229, 0.224, 0.225])
    x1 = face[0]
    y1 = face[1]
    x2 = face[2]
    y2 = face[3]
    height, width, _ = img.shape
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    size = int(min([w, h]) * 1.2)
    cx = x1 + w // 2
    cy = y1 + h // 2
    x1 = cx - size // 2
    x2 = x1 + size
    y1 = cy - size // 2
    y2 = y1 + size

    dx = max(0, -x1)
    dy = max(0, -y1)
    x1 = max(0, x1)
    y1 = max(0, y1)

    edx = max(0, x2 - width)
    edy = max(0, y2 - height)
    x2 = min(width, x2)
    y2 = min(height, y2)
    new_bbox = torch.Tensor([x1, y1, x2, y2]).int()
    x1, y1, x2, y2 = new_bbox
    cropped = img[y1:y2, x1:x2]
    if dx > 0 or dy > 0 or edx > 0 or edy > 0:
        cropped = cv2.copyMakeBorder(
            cropped, int(dy), int(edy), int(dx), int(edx), cv2.BORDER_CONSTANT, 0
        )
    cropped_face = cv2.resize(cropped, (out_size, out_size))

    if cropped_face.shape[0] <= 0 or cropped_face.shape[1] <= 0:
        raise NotADirectoryError
    test_face = cropped_face.copy()
    test_face = test_face / 255.0
    if backbone == "MobileNet":
        test_face = (test_face - mean) / std
    test_face = test_face.transpose((2, 0, 1))
    test_face = torch.from_numpy(test_face).float()
    return dict(data=test_face, bbox=new_bbox)


@torch.no_grad()
def single_predict(model, feed, device):
    landmark = model(feed["data"].unsqueeze(0).to(device)).cpu()
    landmark = landmark.reshape(-1, 2)
    landmark = reproject(feed["bbox"], landmark)
    return landmark.numpy()


@torch.no_grad()
def batch_predict(model, feeds, device):
    if not isinstance(feeds, list):
        feeds = [feeds]
    # loader = DataLoader(FeedDataset(feeds), batch_size=50, shuffle=False)
    data = []
    for feed in feeds:
        data.append(feed["data"].unsqueeze(0))
    data = torch.cat(data, 0).to(device)
    results = []

    landmarks = model(data).cpu()
    for landmark, feed in zip(landmarks, feeds):
        landmark = landmark.reshape(-1, 2)
        landmark = reproject(feed["bbox"], landmark)
        results.append(landmark.numpy())
    return results


@torch.no_grad()
def batch_predict_with_loader(model, feeds, device, batch_size=None):
    if not isinstance(feeds, list):
        feeds = [feeds]
    if len(feeds) == 0:
        return []
    if batch_size is None:
        batch_size = len(feeds)
    
    loader = DataLoader(feeds, batch_size=batch_size, shuffle=False)
    results = []
    for feed in loader:
        landmarks = model(feed["data"].to(device)).cpu()
        for landmark, bbox in zip(landmarks, feed["bbox"]):
            landmark = landmark.reshape(-1, 2)
            landmark = reproject(bbox, landmark)
            results.append(landmark.numpy())
    return results


def split_feeds(all_feeds, all_faces):
    counts = [len(faces) for faces in all_faces]
    sum_now = 0
    ends = [0]
    for i in range(len(counts)):
        sum_now += counts[i]
        end = sum_now
        ends.append(end)
    return [all_feeds[ends[i - 1] : ends[i]] for i in range(1, len(ends))]


@torch.no_grad()
def batch_predict_onnx(ort_session, feeds, batch_size=None):
    if not isinstance(feeds, list):
        feeds = [feeds]
    if batch_size is None:
        batch_size = len(feeds)
    results = []
    for feed in feeds:
        ort_inputs = {
            ort_session.get_inputs()[0].name: to_numpy(feed["data"])[
                None,
            ]
        }
        landmark = ort_session.run(None, ort_inputs)[0][0]
        bbox = feed["bbox"]
        landmark = landmark.reshape(-1, 2)
        landmark = reproject(bbox, landmark)
        results.append(landmark)
    return results

def flatten(l):
    return [item for sublist in l for item in sublist]


def partition(images, size):
    """
    Returns a new list with elements
    of which is a list of certain size.

        >>> partition([1, 2, 3, 4], 3)
        [[1, 2, 3], [4]]
    """
    return [
        images[i : i + size] if i + size <= len(images) else images[i:]
        for i in range(0, len(images), size)
    ]

class LandmarkPredictor:
    def __init__(self, gpu_id=0, backbone="MobileNet", file=None):
        """
        gpu_id: -1 for cpu, onnx for onnx
        """
        if gpu_id == "onnx":
            self._initialize_onnx(backbone, file)
        else:
            self._initialize_normal(gpu_id, backbone, file)

    def _initialize_onnx(self, backbone, file):
        self.device = "onnx"
        if onnx is None:
            raise "Please install onnx and onnx runtime first"
        self.backbone = backbone
        if file is None:
            file = get_default_onnx_file(backbone)
        onnx_model = onnx.load(file)
        onnx.checker.check_model(onnx_model)
        self.model = onnxruntime.InferenceSession(file)
        self._batch_predict = batch_predict_onnx

    def _initialize_normal(self, gpu_id=0, backbone="MobileNet", file=None):
        self.device = get_device(gpu_id)
        self.backbone = backbone
        if backbone == "MobileNet":
            model = MobileNet_GDConv(136)
        elif backbone == "PFLD":
            model = PFLDInference()
        else:
            raise NotImplementedError(backbone)

        weights = load_weights(file, backbone)

        model.load_state_dict(weights)
        self.model = model.to(self.device).eval()

        self._batch_predict = batch_predict_with_loader

    def __call__(self, all_boxes, all_images, from_fd=False, chunk_size=None):
        batch = not is_image(all_images)
        if from_fd:
            all_boxes = detection_adapter(all_boxes, batch=batch)
        if not batch:  # 说明是 1 张图
            if is_box(all_boxes):  # 单张图 单个box
                assert is_image(all_images)
                return self._inner_predict(self.prepare_feed(all_images, all_boxes))
            else:
                feeds = [self.prepare_feed(all_images, box) for box in all_boxes]
                return self._inner_predict(feeds)  # 一张图 多个box
        else:
            assert len(all_boxes) == len(all_images)
            assert is_image(all_images[0])
            # 多张图 多个box列表
            if chunk_size is not None:
                assert isinstance(chunk_size, int)
                image_partitions = partition(all_images, chunk_size)
                box_partitions = partition(all_boxes, chunk_size)
                all_results = [self.batch_predict(boxes, images) for boxes, images in zip(box_partitions, image_partitions)]
                return flatten(all_results)
            else:
                return self.batch_predict(all_boxes, all_images)

    def _inner_predict(self, feeds):
        results = self._batch_predict(self.model, feeds, self.device)
        if not isinstance(feeds, list):
            results = results[0]
        return results

    def batch_predict(self, all_boxes, all_images):
        all_feeds = []
        for i, (faces, image) in enumerate(zip(all_boxes, all_images)):
            feeds = [self.prepare_feed(image, box) for box in faces]
            all_feeds.extend(feeds)
        all_results = self._inner_predict(all_feeds)
        all_results = split_feeds(all_results, all_boxes)
        return all_results

    def prepare_feed(self, img, face):
        return prepare_feed(img, face, self.backbone)
