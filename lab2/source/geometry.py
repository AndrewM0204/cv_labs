import cv2
import numpy as np

from . import config


def good_homography(M):
    if M is None or M.shape != (3, 3):
        return False

    if not np.all(np.isfinite(M)):
        return False

    det = np.linalg.det(M)
    return not (det <= 1e-06 or det > 1_000_000.0)


def reasonable_quad(corners, min_angle=config.MIN_QUAD_ANGLE):
    pts = np.round(corners.reshape(-1, 2)).astype(np.int32)

    if len(pts) != 4:
        return False

    if not cv2.isContourConvex(pts.reshape(-1, 1, 2)):
        return False

    pf = pts.astype(np.float32)

    for i in range(4):
        v1 = pf[(i - 1) % 4] - pf[i]
        v2 = pf[(i + 1) % 4] - pf[i]

        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)

        if n1 < 1e-6 or n2 < 1e-6:
            return False

        cos = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        angle = np.arccos(cos) * 180.0 / np.pi

        if angle < min_angle or angle > 180.0 - min_angle:
            return False

    return True


def valid_box(corners, frame_shape, min_area=15):
    if corners is None:
        return False

    pts = corners.reshape(-1, 2).astype(np.float32)

    if not np.all(np.isfinite(pts)):
        return False

    h, w = frame_shape[:2]
    mx = w * 0.5
    my = h * 0.5

    if np.any(pts[:, 0] < -mx) or np.any(pts[:, 0] > w + mx):
        return False

    if np.any(pts[:, 1] < -my) or np.any(pts[:, 1] > h + my):
        return False

    area = cv2.contourArea(pts)

    if area < min_area or area > 3.0 * w * h:
        return False

    _, _, bw, bh = cv2.boundingRect(pts)
    return bw >= 4 and bh >= 4


def _side_lengths(pts):
    return (
        float(np.linalg.norm(pts[1] - pts[0])),
        float(np.linalg.norm(pts[2] - pts[1])),
        float(np.linalg.norm(pts[3] - pts[2])),
        float(np.linalg.norm(pts[0] - pts[3])),
    )


def plausible_aspect(corners, template_aspect, max_factor=config.MAX_ASPECT_DEVIATION):
    pts = corners.reshape(-1, 2).astype(np.float32)
    s01, s12, s23, s30 = _side_lengths(pts)

    if s12 < 0.5 or s30 < 0.5:
        return False

    avg_h = (s01 + s23) / 2.0
    avg_v = (s12 + s30) / 2.0

    if avg_v < 0.5:
        return False

    ratio = (avg_h / avg_v) / template_aspect
    return 1.0 / max_factor <= ratio <= max_factor


def side_lengths_consistent(corners, max_dev=config.MAX_OPPOSITE_SIDE_DEV):
    pts = corners.reshape(-1, 2).astype(np.float32)
    s01, s12, s23, s30 = _side_lengths(pts)

    if min(s01, s12, s23, s30) < 0.5:
        return False

    if max(s01, s23) / min(s01, s23) > max_dev:
        return False

    return not max(s12, s30) / min(s12, s30) > max_dev


def inliers_cover_quad(corners, inlier_pts, min_coverage=config.INLIER_COVERAGE_MIN):
    if inlier_pts is None or len(inlier_pts) < 4:
        return False

    quad_area = abs(cv2.contourArea(corners.reshape(-1, 2).astype(np.float32)))

    if quad_area < 1:
        return False

    hull = cv2.convexHull(inlier_pts.reshape(-1, 1, 2).astype(np.float32))
    hull_area = abs(cv2.contourArea(hull))

    return hull_area / quad_area >= min_coverage


def shrink_quad(corners, inlier_pts, target_ratio=config.SHRINK_TARGET_RATIO, max_iter=4):
    pts = corners.reshape(-1, 2).astype(np.float32).copy()

    if inlier_pts is None or len(inlier_pts) < 4:
        return pts

    inl = inlier_pts.reshape(-1, 2).astype(np.float32)
    center = pts.mean(0)

    for _ in range(max_iter):
        diag = float(np.linalg.norm(pts[2] - pts[0]))

        if diag < 1:
            return pts

        allowed = target_ratio * diag
        worst = max(float(np.min(np.linalg.norm(inl - c, axis=1))) for c in pts)

        if worst <= allowed:
            return pts

        shrink = max(0.7, min(0.95, allowed / worst))
        pts = center + (pts - center) * shrink

    return pts


def plausible_scale(
    corners,
    template_size,
    min_scale=config.MIN_SCALE,
    max_scale=config.MAX_SCALE,
):
    tw, th = template_size
    pts = corners.reshape(-1, 2).astype(np.float32)

    s01, s12, s23, s30 = _side_lengths(pts)

    w = (s01 + s23) / 2.0
    h = (s12 + s30) / 2.0

    return min_scale <= w / tw <= max_scale and min_scale <= h / th <= max_scale


def boxes_consistent(new_corners, old_corners, lost=0):
    if old_corners is None or new_corners is None:
        return True

    n = new_corners.reshape(-1, 2).astype(np.float32)
    o = old_corners.reshape(-1, 2).astype(np.float32)

    if n.shape != (4, 2) or o.shape != (4, 2):
        return False

    nd = float(np.linalg.norm(n[2] - n[0]))
    od = float(np.linalg.norm(o[2] - o[0]))

    if od < 1e-6 or nd < 1e-6:
        return False

    move = float(np.linalg.norm(n.mean(0) - o.mean(0)))

    if move > config.MAX_BOX_JUMP_RATIO * (1 + lost * 2) * max(nd, od):
        return False

    size_limit = config.MAX_BOX_SIZE_CHANGE * (1.0 + 0.5 * lost)
    r = nd / od

    return not (r > size_limit or r < 1.0 / size_limit)


def object_size_from_corners(corners):
    if corners is None:
        return 0

    _, _, bw, bh = cv2.boundingRect(np.int32(corners).reshape(-1, 2))
    return min(bw, bh)


def expand_box(corners, frame_shape, margin=config.ROI_MARGIN, extra=0.0):
    x, y, w, h = cv2.boundingRect(np.int32(corners).reshape(-1, 2))
    H, W = frame_shape[:2]

    obj = max(w, h)

    if obj < 80:
        mx = int(max(w * (1.0 + extra), 48 * (1.0 + extra)))
        my = int(max(h * (1.0 + extra), 48 * (1.0 + extra)))
    else:
        eff = margin + extra
        mx = int(max(w * eff, 24))
        my = int(max(h * eff, 24))

    x0 = max(0, x - mx)
    y0 = max(0, y - my)
    x1 = min(W, x + w + mx)
    y1 = min(H, y + h + my)

    return x0, y0, x1 - x0, y1 - y0