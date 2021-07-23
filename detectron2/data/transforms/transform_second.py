import random
import numpy as np
import cv2


class DotaTransformTrain:
    def __init__(self, width: int, height: int, scale_factor: int) -> None:
        self._width = width
        self._height = height
        self._scale_factor = scale_factor

    def __call__(self, result):
        print(result["img_shape"])

        h, w, _ = result["img_shape"]
        input_h, input_w = self._height, self._width
        s = max(h, w) * 1.0
        c = np.array([w / 2.0, h / 2.0], dtype=np.float32)
        sf = 0.4
        w_border = get_border(128, w)
        h_border = get_border(128, h)
        c[0] = np.random.randint(low=w_border, high=w - w_border)
        c[1] = np.random.randint(low=h_border, high=h - h_border)
        s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)

        trans_input = get_affine_transform(c, s, 0, [input_w, input_h])
        inp = cv2.warpAffine(
            result["img"], trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR
        )
        output_w = input_w // self._scale_factor
        output_h = input_h // self._scale_factor
        trans_output = get_affine_transform(c, s, 0, [output_w, output_h])

        bbox = result["gt_bboxes"]

        for i in range(bbox.shape[0]):
            bbox[i, :2] = affine_transform(bbox[i, :2], trans_output)
            bbox[i, 2:4] = affine_transform(bbox[i, 2:4], trans_output)
            bbox[i, 4:6] = affine_transform(bbox[i, 4:6], trans_output)
            bbox[i, 6:8] = affine_transform(bbox[i, 6:8], trans_output)
        bbox[:, :8] = np.clip(bbox[:, :8], 0, output_w - 1)
        img = inp
        result["img"] = img

        gt_bboxes = minAreaRect(bbox[:, :8])
        gt_ids = bbox[:, 8:9]
        result["gt_bboxes"] = gt_bboxes
        result["gt_labels"] = gt_ids
        return result


def minAreaRect(points: np.ndarray, angle_type: int = 180) -> np.ndarray:
    result = []
    for line in points:
        bbox_np = np.int0(line).reshape((4, 2))
        rect = cv2.minAreaRect(bbox_np)
        x, y = rect[0][0], rect[0][1]
        w, h, theta = rect[1][0], rect[1][1], rect[2]
        if angle_type == 180:
            if w < h:
                w, h = h, w
                theta += 90
        theta = (theta / 180) * np.pi
        result.append(np.array([x, y, w, h, theta]))
    return np.array(result)


def get_border(border, size):
    """Get the border size of the image"""
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i


def get_affine_transform(
    center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=0
):
    """Get affine transform matrix given center, scale and rotation.

    Parameters
    ----------
    center : tuple of float
        Center point.
    scale : float
        Scaling factor.
    rot : float
        Rotation degree.
    output_size : tuple of int
        (width, height) of the output size.
    shift : float
        Shift factor.
    inv : bool
        Whether inverse the computation.

    Returns
    -------
    numpy.ndarray
        Affine matrix.

    """

    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_rot_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    return trans


def affine_transform(pt, t):
    """Apply affine transform to a bounding box given transform matrix t.

    Parameters
    ----------
    pt : numpy.ndarray
        Bounding box with shape (1, 2).
    t : numpy.ndarray
        Transformation matrix with shape (2, 3).

    Returns
    -------
    numpy.ndarray
        New bounding box with shape (1, 2).

    """
    new_pt = np.array([pt[0], pt[1], 1.0], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_rot_dir(src_point, rot_rad):
    """Get rotation direction.

    Parameters
    ----------
    src_point : tuple of float
        Original point.
    rot_rad : float
        Rotation radian.

    Returns
    -------
    tuple of float
        Rotation.

    """
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_3rd_point(a, b):
    """Get the 3rd point position given first two points.

    Parameters
    ----------
    a : tuple of float
        First point.
    b : tuple of float
        Second point.

    Returns
    -------
    tuple of float
        Third point.

    """
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


_data_rng = np.random.RandomState(None)


def np_random_color_distort(
    image, data_rng=None, eig_val=None, eig_vec=None, var=0.4, alphastd=0.1
):
    """Numpy version of random color jitter.

    Parameters
    ----------
    image : numpy.ndarray
        original image.
    data_rng : numpy.random.rng
        Numpy random number generator.
    eig_val : numpy.ndarray
        Eigen values.
    eig_vec : numpy.ndarray
        Eigen vectors.
    var : float
        Variance for the color jitters.
    alphastd : type
        Jitter for the brightness.

    Returns
    -------
    numpy.ndarray
        The jittered image

    """
    if data_rng is None:
        data_rng = _data_rng
    if eig_val is None:
        eig_val = np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32)
    if eig_vec is None:
        eig_vec = np.array(
            [
                [-0.58752847, -0.69563484, 0.41340352],
                [-0.5832747, 0.00994535, -0.81221408],
                [-0.56089297, 0.71832671, 0.41158938],
            ],
            dtype=np.float32,
        )

    def grayscale(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def lighting_(data_rng, image, alphastd, eigval, eigvec):
        alpha = data_rng.normal(scale=alphastd, size=(3,))
        image += np.dot(eigvec, eigval * alpha)

    def blend_(alpha, image1, image2):
        image1 *= alpha
        image2 *= 1 - alpha
        image1 += image2

    def saturation_(data_rng, image, gs, gs_mean, var):
        # pylint: disable=unused-argument
        alpha = 1.0 + data_rng.uniform(low=-var, high=var)
        blend_(alpha, image, gs[:, :, None])

    def brightness_(data_rng, image, gs, gs_mean, var):
        # pylint: disable=unused-argument
        alpha = 1.0 + data_rng.uniform(low=-var, high=var)
        image *= alpha

    def contrast_(data_rng, image, gs, gs_mean, var):
        # pylint: disable=unused-argument
        alpha = 1.0 + data_rng.uniform(low=-var, high=var)
        blend_(alpha, image, gs_mean)

    functions = [brightness_, contrast_, saturation_]
    random.shuffle(functions)

    gs = grayscale(image)
    gs_mean = gs.mean()
    for f in functions:
        f(data_rng, image, gs, gs_mean, var)
    lighting_(data_rng, image, alphastd, eig_val, eig_vec)
    return image
