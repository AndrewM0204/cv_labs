import cv2
import numpy as np

from . import config, geometry


def select_lk_points_for_ref(ref_entry, max_pts=config.MAX_LK_POINTS):
    if ref_entry is None:
        return None

    kp = ref_entry["kp"]
    scale = ref_entry["scale"]

    if len(kp) < 6:
        return None

    pts = np.float32([k.pt for k in kp]).reshape(-1, 1, 2) / float(scale)

    if len(kp) > max_pts:
        responses = np.array([k.response for k in kp])
        idx = np.argsort(-responses)[:max_pts]
        pts = pts[idx]

    return pts


def get_lk_params(obj_size):
    if obj_size < config.MIN_LK_SIZE:
        return None

    if obj_size < 80:
        win, lvl = 7, 2
    elif obj_size < 200:
        win, lvl = 11, 2
    elif obj_size < 400:
        win, lvl = 15, 3
    else:
        win, lvl = 21, 3

    return {
        "winSize": (win, win),
        "maxLevel": lvl,
        "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    }


def track_flow(prev_gray, gray, src_pts, prev_pts, lk_params):
    if src_pts is None or prev_pts is None or len(src_pts) < 6:
        return None, 0, None, None

    try:
        cur_pts, st, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            gray,
            prev_pts,
            None,
            **lk_params
        )

        back_pts, st2, _ = cv2.calcOpticalFlowPyrLK(
            gray,
            prev_gray,
            cur_pts,
            None,
            **lk_params
        )
    except cv2.error:
        return None, 0, None, None

    if cur_pts is None or st is None or back_pts is None or st2 is None:
        return None, 0, None, None

    fb = np.linalg.norm(
        prev_pts.reshape(-1, 2) - back_pts.reshape(-1, 2),
        axis=1,
    )

    good = (st.flatten() == 1) & (st2.flatten() == 1) & (fb < config.FB_THRESHOLD)

    if good.sum() < 6:
        return None, 0, None, None

    src = src_pts.reshape(-1, 2)[good]
    dst = cur_pts.reshape(-1, 2)[good]

    M, mask = cv2.findHomography(
        src.reshape(-1, 1, 2),
        dst.reshape(-1, 1, 2),
        cv2.USAC_MAGSAC,
        2.0,
        maxIters=2000,
        confidence=0.995,
    )

    if not geometry.good_homography(M):
        return None, 0, None, None

    inl_mask = mask.ravel() == 1 if mask is not None else np.ones(len(dst), bool)

    return M, int(inl_mask.sum()), cur_pts.reshape(-1, 2), dst[inl_mask]


def accept_lk_result(
    corners,
    inlier_pts,
    frame_shape,
    template_aspect,
    template_size,
    last_corners,
    lost,
):
    if not (geometry.valid_box(corners, frame_shape) and geometry.reasonable_quad(corners)):
        return False

    if not geometry.plausible_scale(corners, template_size):
        return False

    if not geometry.side_lengths_consistent(corners):
        return False

    return geometry.boxes_consistent(corners.reshape(-1, 2), last_corners, lost)