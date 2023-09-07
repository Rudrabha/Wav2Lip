import cv2
import numpy as np
import torch

try:
    from torch.utils.model_zoo import download_url_to_file
except ImportError:
    from torch.hub import download_url_to_file
import errno
import sys
import os
import warnings
import re
from urllib.parse import urlparse

ENV_TORCH_HOME = "TORCH_HOME"
ENV_XDG_CACHE_HOME = "XDG_CACHE_HOME"
DEFAULT_CACHE_DIR = "~/.cache"

# matches bfd8deac from resnet18-bfd8deac.pth
HASH_REGEX = re.compile(r"-([a-f0-9]*)\.")


def drawLandmark_multiple(img, bbox=None, landmark=None, color=(0, 255, 0)):
    """
    Input:
    - img: gray or RGB
    - bbox: type of BBox
    - landmark: reproject landmark of (5L, 2L)
    Output:
    - img marked with landmark and bbox
    """
    img = cv2.UMat(img).get()
    if bbox is not None:
        x1, y1, x2, y2 = np.array(bbox)[:4].astype(np.int32)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    if landmark is not None:
        for x, y in np.array(landmark).astype(np.int32):
            cv2.circle(img, (int(x), int(y)), 2, color, -1)
    return img


draw_landmarks=drawLandmark_multiple

pretrained_urls = {
    "MobileNet": "https://github.com/elliottzheng/fast-alignment/releases/download/weights_v1/mobilenet_224_model_best_gdconv_external.pth",
    "PFLD": "https://github.com/elliottzheng/fast-alignment/releases/download/weights_v1/pfld_model_best.pth",
    "PFLD_onnx": "https://github.com/elliottzheng/fast-alignment/releases/download/weights_v1/PFLD.onnx",
}


def load_weights(file, backbone):
    if file is None:
        assert backbone in pretrained_urls
        url = pretrained_urls[backbone]
        return torch.utils.model_zoo.load_url(url)
    else:
        return torch.load(file, map_location="cpu")


def _get_torch_home():
    torch_home = os.path.expanduser(
        os.getenv(
            ENV_TORCH_HOME,
            os.path.join(os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR), "torch"),
        )
    )
    return torch_home


def auto_download_from_url(url, model_dir=None, map_location=None, progress=True):
    r"""Loads the Torch serialized object at the given URL.
    If the object is already present in `model_dir`, it's deserialized and
    returned. The filename part of the URL should follow the naming convention
    ``filename-<sha256>.ext`` where ``<sha256>`` is the first eight or more
    digits of the SHA256 hash of the contents of the file. The hash is used to
    ensure unique names and to verify the contents of the file.
    The default value of `model_dir` is ``$TORCH_HOME/checkpoints`` where
    environment variable ``$TORCH_HOME`` defaults to ``$XDG_CACHE_HOME/torch``.
    ``$XDG_CACHE_HOME`` follows the X Design Group specification of the Linux
    filesytem layout, with a default value ``~/.cache`` if not set.
    Args:
        url (string): URL of the object to download
        model_dir (string, optional): directory in which to save the object
        map_location (optional): a function or a dict specifying how to remap storage locations (see torch.load)
        progress (bool, optional): whether or not to display a progress bar to stderr
    Example:
        >>> state_dict = torch.hub.load_state_dict_from_url('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth')
    """
    # Issue warning to move data if old env is set
    if os.getenv("TORCH_MODEL_ZOO"):
        warnings.warn(
            "TORCH_MODEL_ZOO is deprecated, please use env TORCH_HOME instead"
        )

    if model_dir is None:
        torch_home = _get_torch_home()
        model_dir = os.path.join(torch_home, "checkpoints")

    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Directory already exists, ignore.
            pass
        else:
            # Unexpected OSError, re-raise.
            raise

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        download_url_to_file(url, cached_file, None, progress=progress)
    return cached_file


def get_default_onnx_file(backbone):
    key = backbone + "_onnx"
    if key not in pretrained_urls:
        raise "default checkpoint for %s is not available" % backbone
    return auto_download_from_url(pretrained_urls[key])


def is_image(x):
    if isinstance(x, np.ndarray) and len(x.shape) == 3 and x.shape[-1] == 3:
        return True
    else:
        return False


def is_box(x):
    try:
        x = np.array(x)
        assert len(x) == 4
        assert (x[2:] - x[:2]).min() > 0
        return True
    except:
        return False


def is_face(x):
    try:
        assert is_box(x[0])
        return True
    except:
        return False


def detection_adapter(all_faces, batch=False):
    if not batch:
        if is_face(all_faces):  # 单个检测结果
            return all_faces[0]
        else:
            return [face[0] for face in all_faces]  # 是单层列表
    else:
        return [[face[0] for face in faces] for faces in all_faces]  # 双层列表


def to_numpy(tensor):
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


def bbox_from_pts(ldm_new):
    (x1, y1), (x2, y2) = ldm_new.min(0), ldm_new.max(0)
    box_new = np.array([x1, y1, x2, y2])
    box_new[:2] -= 10
    box_new[2:] += 10
    return box_new


class Aligner:
    def __init__(self, standard_points, size) -> None:
        self.standard_points = standard_points  # ndarray of N,2
        self.size = size

    def __call__(self, img, landmarks):
        # ndarray image, landmarks N,2
        from skimage import transform

        trans = transform.SimilarityTransform()
        res = trans.estimate(landmarks, self.standard_points)
        M = trans.params
        new_img = cv2.warpAffine(img, M[:2, :], dsize=(self.size, self.size))
        return new_img
