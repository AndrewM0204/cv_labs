import cv2
import numpy as np

from . import config, geometry


def make_sift():
    return cv2.SIFT_create(
        nfeatures=0,
        nOctaveLayers=3,
        contrastThreshold=0.02,
        edgeThreshold=10,
        sigma=1.6,
    )


def match_descriptors(flann, des_ref, des_cur, ratio):
    if des_ref is None or des_cur is None:
        return []

    if len(des_ref) < 2 or len(des_cur) < 2:
        return []

    try:
        fwd = flann.knnMatch(des_ref, des_cur, k=2)
        bwd = flann.knnMatch(des_cur, des_ref, k=2)
    except cv2.error:
        return []

    good_fwd = [
        m
        for m, n in (p for p in fwd if len(p) == 2)
        if m.distance < ratio * n.distance
    ]

    good_bwd = {
        m.queryIdx
        for m, n in (p for p in bwd if len(p) == 2)
        if m.distance < ratio * n.distance
    }

    return [m for m in good_fwd if m.trainIdx in good_bwd]


def build_ref_pyramid(gray_ref, sift, scales):
    pyramid = []

    scales = [min(1.0, float(s)) for s in scales if float(s) > 0]
    all_scales = sorted(set([1.0] + scales), reverse=True)

    for s in all_scales:
        if s >= 1.0 - 1e-6:
            g = gray_ref
            s = 1.0
        else:
            nw = max(16, round(gray_ref.shape[1] * s))
            nh = max(16, round(gray_ref.shape[0] * s))
            g = cv2.resize(gray_ref, (nw, nh), interpolation=cv2.INTER_AREA)

        kp, des = sift.detectAndCompute(g, None)

        if des is None or len(kp) < 4:
            continue

        corners = np.float32(
            [
                [0, 0],
                [g.shape[1], 0],
                [g.shape[1], g.shape[0]],
                [0, g.shape[0]],
            ]
        ).reshape(-1, 1, 2)

        pyramid.append(
            {
                "scale": s,
                "kp": kp,
                "des": des,
                "corners": corners,
            }
        )

    return pyramid


def _match_pyramid(
    ref_pyramid,
    kp_cur,
    des_cur,
    flann,
    ratio,
    min_matches,
    min_inliers,
    transform,
    validator,
    template_aspect,
    lost=0,
):
    ratio_use = min(0.85, ratio + 0.02 * lost)

    aspect_factor = config.MAX_ASPECT_DEVIATION * (1.0 + 0.10 * lost)
    side_factor = config.MAX_OPPOSITE_SIDE_DEV * (1.0 + 0.10 * lost)
    coverage_min = max(0.05, config.INLIER_COVERAGE_MIN - 0.01 * lost)

    best = None

    for ref in ref_pyramid:
        required = max(
            4,
            round(min_matches * max(0.45, ref["scale"] ** 0.3))
        )

        matches = match_descriptors(flann, ref["des"], des_cur, ratio_use)

        if len(matches) < required:
            continue

        src = np.float32(
            [ref["kp"][m.queryIdx].pt for m in matches]
        ).reshape(-1, 1, 2)

        dst = np.float32(
            [kp_cur[m.trainIdx].pt for m in matches]
        ).reshape(-1, 1, 2)

        thresh = max(1.5, 3.0 * ref["scale"])

        M, mask = cv2.findHomography(
            src,
            dst,
            cv2.USAC_MAGSAC,
            thresh,
            maxIters=2000,
            confidence=0.995,
        )

        if not geometry.good_homography(M):
            continue

        try:
            corners_frame = transform(cv2.perspectiveTransform(ref["corners"], M))
        except cv2.error:
            continue

        score = int(mask.sum()) if mask is not None else 0
        min_score = max(4, min(min_inliers, required))

        if score < min_score:
            continue

        inl_mask = mask.ravel() == 1 if mask is not None else np.ones(len(dst), bool)

        if inl_mask.sum() < min_score:
            continue

        inlier_pts = transform(dst.reshape(-1, 2)[inl_mask])

        corners_2d = geometry.shrink_quad(corners_frame.reshape(-1, 2), inlier_pts)
        corners_frame = corners_2d.reshape(-1, 1, 2)

        if not validator(corners_frame):
            continue

        if not geometry.plausible_aspect(corners_frame, template_aspect, aspect_factor):
            continue

        if not geometry.side_lengths_consistent(corners_frame, side_factor):
            continue

        if not geometry.inliers_cover_quad(corners_frame, inlier_pts, coverage_min):
            continue

        if best is None or score > best[1]:
            pts_frame = transform(dst.reshape(-1, 2))
            best = (corners_frame, score, ref, pts_frame, mask)

    return best


def detect_full_frame(
    ref_pyramid,
    gray,
    sift,
    flann,
    ratio,
    min_matches,
    template_aspect,
    template_size,
    last_size=0,
    lost=0,
):
    h, w = gray.shape[:2]

    if lost > 0 or last_size < 60:
        max_dim = config.FULL_FRAME_MAX_DIM_SMALL_OBJ
    elif last_size >= 120:
        max_dim = config.FULL_FRAME_MAX_DIM_LARGE_OBJ
    else:
        max_dim = config.FULL_FRAME_MAX_DIM

    scale = min(1.0, max_dim / float(max(h, w)))

    if scale < 1.0:
        small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        small = gray
        scale = 1.0

    kp_cur, des_cur = sift.detectAndCompute(small, None)

    if des_cur is None or len(kp_cur) < 2:
        return None, 0, None, None, None

    def transform(pts):
        return pts / scale

    def validator(c):
        return (
            geometry.valid_box(c, gray.shape)
            and geometry.reasonable_quad(c)
            and geometry.plausible_scale(c, template_size)
        )

    best = _match_pyramid(
        ref_pyramid,
        kp_cur,
        des_cur,
        flann,
        ratio,
        min_matches,
        config.FULL_FRAME_MIN_INLIERS,
        transform=transform,
        validator=validator,
        template_aspect=template_aspect,
        lost=lost,
    )

    return best if best is not None else (None, 0, None, None, None)


def detect_in_roi(
    ref_pyramid,
    gray,
    last_corners,
    sift,
    flann,
    ratio,
    min_matches,
    lost,
    template_aspect,
    template_size,
):
    extra = min(lost * 0.25, 1.25)
    x, y, w, h = geometry.expand_box(last_corners, gray.shape, extra=extra)

    if w < config.MIN_ROI_PX or h < config.MIN_ROI_PX:
        return None, 0, None, None, None

    roi = gray[y:y + h, x:x + w]

    if roi.size == 0:
        return None, 0, None, None, None

    scale = 1.0
    max_side = max(w, h)

    if max_side < config.UPSCALE_TARGET_PX:
        scale = min(config.MAX_UPSCALE, config.UPSCALE_TARGET_PX / float(max_side))
        interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_LINEAR
        roi = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=interpolation)
    elif max_side > config.MAX_ROI_PX:
        scale = config.MAX_ROI_PX / float(max_side)
        roi = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    kp_cur, des_cur = sift.detectAndCompute(roi, None)

    if des_cur is None or len(kp_cur) < 2:
        return None, 0, None, None, None

    ox = x
    oy = y
    sc = scale

    def transform(pts):
        return pts / sc + np.float32([ox, oy])

    def validator(c):
        return (
            geometry.valid_box(c, gray.shape)
            and geometry.reasonable_quad(c)
            and geometry.plausible_scale(c, template_size)
        )

    best = _match_pyramid(
        ref_pyramid,
        kp_cur,
        des_cur,
        flann,
        ratio,
        min_matches,
        max(6, min_matches // 2),
        transform=transform,
        validator=validator,
        template_aspect=template_aspect,
        lost=lost,
    )

    return best if best is not None else (None, 0, None, None, None)