import torch
import numpy as np
from .params import *
from torchvision import transforms
from batch_face.utils import auto_download_from_url

STD_SIZE = 120


pretrained_urls = {
    "3DDFA": "https://github.com/elliottzheng/fast-alignment/releases/download/weights_v1/phase1_wpdc_vdc.pth.tar"
}


def get_default_fr_file(backend):
    return auto_download_from_url(pretrained_urls[backend])


def clip(center, size, max_size):
    end = center + size / 2
    if end > max_size:
        end = max_size
    start = end - size
    if start < 0:
        start = 0
        end = start + size
    return start, end


def parse_roi_box_from_bbox(bbox, img_shape):
    h, w = img_shape
    left, top, right, bottom = bbox
    old_size = (right - left + bottom - top) / 2
    center_x = right - (right - left) / 2.0
    center_y = bottom - (bottom - top) / 2.0 + old_size * 0.14
    size = int(old_size * 1.58)

    roi_box = np.zeros((4))
    roi_box[[0, 2]] = clip(center_x, size, w)
    roi_box[[1, 3]] = clip(center_y, size, h)
    return roi_box


def parse_param(param):
    """Work for both numpy and tensor"""
    p_ = param[:12].reshape(3, -1)
    p = p_[:, :3]
    offset = p_[:, -1].reshape(3, 1)
    alpha_shp = param[12:52].reshape(-1, 1)
    alpha_exp = param[52:].reshape(-1, 1)
    return p, offset, alpha_shp, alpha_exp


def reconstruct_vertex(param, whitening=True, dense=False, transform=True):
    """Whitening param -> 3d vertex, based on the 3dmm param: u_base, w_shp, w_exp
    dense: if True, return dense vertex, else return 68 sparse landmarks. All dense or sparse vertex is transformed to
    image coordinate space, but without alignment caused by face cropping.
    transform: whether transform to image space
    """
    if len(param) == 12:
        param = np.concatenate((param, [0] * 50))
    if whitening:
        if len(param) == 62:
            param = param * param_std + param_mean
        else:
            param = np.concatenate((param[:11], [0], param[11:]))
            param = param * param_std + param_mean

    p, offset, alpha_shp, alpha_exp = parse_param(param)

    if dense:
        vertex = (
            p @ (u + w_shp @ alpha_shp + w_exp @ alpha_exp).reshape(3, -1, order="F")
            + offset
        )

        if transform:
            # transform to image coordinate space
            vertex[1, :] = std_size + 1 - vertex[1, :]
    else:
        """For 68 pts"""
        vertex = (
            p
            @ (u_base + w_shp_base @ alpha_shp + w_exp_base @ alpha_exp).reshape(
                3, -1, order="F"
            )
            + offset
        )

        if transform:
            # transform to image coordinate space
            vertex[1, :] = std_size + 1 - vertex[1, :]

    return vertex


def _predict_vertices(param, roi_bbox, dense, transform=True):
    vertex = reconstruct_vertex(param, dense=dense)
    sx, sy, ex, ey = roi_bbox
    scale_x = (ex - sx) / 120
    scale_y = (ey - sy) / 120
    vertex[0, :] = vertex[0, :] * scale_x + sx
    vertex[1, :] = vertex[1, :] * scale_y + sy

    s = (scale_x + scale_y) / 2
    vertex[2, :] *= s

    return vertex


def predict_68pts(param, roi_box):
    return _predict_vertices(param, roi_box, dense=False)


def predict_dense(param, roi_box):
    return _predict_vertices(param, roi_box, dense=True)


def crop_img(img, roi_box):
    h, w = img.shape[:2]

    sx, sy, ex, ey = [int(round(_)) for _ in roi_box]

    dh, dw = ey - sy, ex - sx
    if len(img.shape) == 3:
        res = np.zeros((dh, dw, 3), dtype=np.uint8)
    else:
        res = np.zeros((dh, dw), dtype=np.uint8)
    if sx < 0:
        sx, dsx = 0, -sx
    else:
        dsx = 0

    if ex > w:
        ex, dex = w, dw - (ex - w)
    else:
        dex = dw

    if sy < 0:
        sy, dsy = 0, -sy
    else:
        dsy = 0

    if ey > h:
        ey, dey = h, dh - (ey - h)
    else:
        dey = dh

    res[dsy:dey, dsx:dex] = img[sy:ey, sx:ex]
    return res


class ToTensorGjz(object):
    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img.float()

    def __repr__(self):
        return self.__class__.__name__ + "()"


class NormalizeGjz(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        tensor.sub_(self.mean).div_(self.std)
        return tensor


transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])
normalize = NormalizeGjz(mean=127.5, std=128)
