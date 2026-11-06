import cv2
import numpy as np

from . import config


def draw_keypoints(frame, pts, color, radius=2, thickness=-1):
    if pts is None or len(pts) == 0:
        return

    for p in pts:
        cv2.circle(
            frame,
            (int(p[0]), int(p[1])),
            radius,
            color,
            thickness,
        )


def draw_tracking(frame, corners, color, label="", pts=None, show_kp=False):
    if corners is None:
        return

    if show_kp and pts is not None:
        draw_keypoints(frame, pts, color)

    pts_int = np.int32(corners).reshape(-1, 1, 2)

    cv2.polylines(frame, [pts_int], True, color, 2)

    bx, by, _, _ = cv2.boundingRect(pts_int)

    if label:
        cv2.putText(
            frame,
            label,
            (bx, max(20, by - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )


def select_template(template_img):
    name = "Select object, then press ENTER or SPACE"

    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    roi = cv2.selectROI(name, template_img, showCrosshair=True)
    cv2.destroyWindow(name)

    if roi is None:
        return template_img, None

    x, y, w, h = map(int, roi)

    if w > 5 and h > 5:
        return template_img[y:y + h, x:x + w].copy(), (x, y, w, h)

    return template_img, None


def window_is_open():
    try:
        return cv2.getWindowProperty(config.WINDOW_NAME, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        return False