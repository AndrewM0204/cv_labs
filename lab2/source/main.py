import argparse
import time

import cv2
import numpy as np

from . import config, drawing, geometry, lk, matching, video_io


def parse_args():
    parser = argparse.ArgumentParser(description="Отслеживание плоского объекта по ключевым точкам")
    parser.add_argument("video", help="путь к видеофайлу")
    parser.add_argument(
        "--output", 
        default=None, 
        help="путь к файлу для сохранения видео с результатом трекинга (например, result.mp4)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or not (fps > 1.0 and fps <= 240.0):
        fps = 30.0

    ret, first_frame = cap.read()
    if not ret:
        return

    template = first_frame
    H0, W0 = first_frame.shape[:2]
    initial_box = np.float32([[0, 0], [W0, 0], [W0, H0], [0, H0]])

    gray_ref = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    h_ref, w_ref = gray_ref.shape[:2]
    template_size = (w_ref, h_ref)
    template_aspect = w_ref / float(h_ref)

    ref_corners = np.float32(
        [
            [0, 0],
            [w_ref, 0],
            [w_ref, h_ref],
            [0, h_ref],
        ]
    ).reshape(-1, 1, 2)

    sift = matching.make_sift()
    pyramid = matching.build_ref_pyramid(gray_ref, sift, list(config.DEFAULT_PYRAMID))

    if not pyramid:
        return

    flann = cv2.FlannBasedMatcher(
        {"algorithm": 1, "trees": 5},
        {"checks": 64},
    )

    use_lk = True
    show_kp = True

    gray_first = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    M_current, _ = cv2.findHomography(
        ref_corners,
        initial_box.reshape(-1, 1, 2).astype(np.float32),
    )

    if not geometry.good_homography(M_current):
        return

    lk_pts = lk.select_lk_points_for_ref(pyramid[0], config.MAX_LK_POINTS)

    if lk_pts is None:
        use_lk = False

    prev_gray = gray_first
    last_corners = initial_box.reshape(-1, 2).astype(np.float32)

    lost = 0
    frames_since_sift = 0
    full_frame_cooldown = 0

    candidate_box = None
    candidate_count = 0

    cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_NORMAL)

    throttle = True
    start_time = time.time()
    frame_idx = 0

    writer = None
    if args.output:
        writer = video_io.create_video_writer(
            args.output,
            fps,
            (first_frame.shape[1], first_frame.shape[0])
        )

    preview = first_frame.copy()
    drawing.draw_tracking(
        preview,
        last_corners,
        (0, 255, 0),
        label="OK",
        pts=None,
        show_kp=False,
    )

    if writer is not None:
        writer.write(preview)

    cv2.imshow(config.WINDOW_NAME, preview)
    cv2.waitKey(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        H, W = frame.shape[:2]

        obj_size = geometry.object_size_from_corners(last_corners)
        frames_since_sift += 1

        if full_frame_cooldown > 0:
            full_frame_cooldown -= 1

        # ---------- 1. LK ----------
        lk_corners = None
        lk_M = None
        lk_inl = 0
        lk_track_pts = None

        lk_params = lk.get_lk_params(obj_size) if use_lk else None

        if (
            lk_params is not None
            and lost == 0
            and M_current is not None
            and lk_pts is not None
            and last_corners is not None
        ):
            try:
                prev_pts = cv2.perspectiveTransform(lk_pts, M_current)

                M_lk, lk_inl, lk_track_pts, lk_inlier_pts = lk.track_flow(
                    prev_gray,
                    gray,
                    lk_pts,
                    prev_pts,
                    lk_params,
                )

                if M_lk is not None:
                    c = cv2.perspectiveTransform(ref_corners, M_lk)

                    if lk.accept_lk_result(
                        c,
                        lk_inlier_pts,
                        frame.shape,
                        template_aspect,
                        template_size,
                        last_corners,
                        lost,
                    ):
                        lk_corners = c.reshape(-1, 2)
                        lk_M = M_lk

            except cv2.error:
                lk_track_pts = None

        # ---------- 2. SIFT throttling ----------
        should_sift = False

        if lk_corners is None:
            if lost == 0:
                should_sift = True
            else:
                should_sift = frames_since_sift >= config.ROI_SIFT_EVERY_LOST
        else:
            area = abs(cv2.contourArea(lk_corners.astype(np.float32)))
            ar = area / float(H * W)

            interval = (
                config.SMALL_OBJ_INTERVAL
                if ar < config.SMALL_OBJ_RATIO
                else config.LARGE_OBJ_INTERVAL
            )

            if frames_since_sift >= interval:
                should_sift = True

        # ---------- 3. SIFT ----------
        sift_corners = None
        sift_score = 0
        sift_ref = None
        sift_pts = None
        sift_mask = None

        if should_sift:
            if last_corners is not None:
                c, s, r, pts, mask = matching.detect_in_roi(
                    pyramid,
                    gray,
                    last_corners,
                    sift,
                    flann,
                    config.DEFAULT_RATIO,
                    config.DEFAULT_MIN_MATCHES,
                    lost,
                    template_aspect,
                    template_size,
                )

                if c is not None and geometry.boxes_consistent(
                    c.reshape(-1, 2),
                    last_corners,
                    lost,
                ):
                    sift_corners, sift_score, sift_ref = c, s, r
                    sift_pts, sift_mask = pts, mask

            if sift_corners is None and lk_corners is None and lost >= 1:
                if full_frame_cooldown <= 0:
                    c, s, r, pts, mask = matching.detect_full_frame(
                        pyramid,
                        gray,
                        sift,
                        flann,
                        config.DEFAULT_RATIO,
                        config.DEFAULT_MIN_MATCHES,
                        template_aspect,
                        template_size,
                        last_size=obj_size,
                        lost=lost,
                    )

                    if c is not None:
                        sift_corners, sift_score, sift_ref = c, s, r
                        sift_pts, sift_mask = pts, mask

                    full_frame_cooldown = config.FULL_FRAME_EVERY

            frames_since_sift = 0

        # ---------- 4. Confirmation ----------
        sift_accepted = False

        if sift_corners is not None:
            cur_pts = sift_corners.reshape(-1, 2)

            confirm_needed = (
                config.CONFIRM_FRAMES_LONG_LOST
                if lost >= config.LONG_LOST_THRESHOLD
                else config.CONFIRM_FRAMES
            )

            if lost == 0:
                sift_accepted = True
                candidate_box = None
                candidate_count = 0
            else:
                if candidate_box is not None and geometry.boxes_consistent(
                    cur_pts,
                    candidate_box,
                    lost=2,
                ):
                    candidate_count += 1
                else:
                    candidate_box = cur_pts.copy()
                    candidate_count = 1

                if candidate_count >= confirm_needed:
                    sift_accepted = True
                    candidate_box = None
                    candidate_count = 0

        # ---------- 5. Choose result ----------
        final = None
        label = "OBJECT"
        found = False

        draw_pts = None
        draw_color = (0, 255, 0)

        if sift_accepted and sift_corners is not None:
            final = sift_corners.reshape(-1, 2)

            M_current, _ = cv2.findHomography(
                ref_corners,
                sift_corners.reshape(-1, 1, 2).astype(np.float32),
            )

            if not geometry.good_homography(M_current):
                M_current = None

            if use_lk and sift_ref is not None:
                new_lk = lk.select_lk_points_for_ref(sift_ref, config.MAX_LK_POINTS)

                if new_lk is not None and len(new_lk) >= 6:
                    lk_pts = new_lk

            lost = 0
            frames_since_sift = 0
            full_frame_cooldown = 0
            candidate_box = None
            candidate_count = 0

            found = True

            draw_pts = sift_pts
            draw_color = (0, 255, 0)

        elif lk_corners is not None:
            final = lk_corners
            M_current = lk_M

            lost = 0
            found = True

            draw_pts = lk_track_pts

        else:
            lost += 1

            if lost > 60:
                last_corners = None
                M_current = None
                candidate_box = None
                candidate_count = 0

            final = None

        if found:
            last_corners = final.copy()

        # ---------- 6. Draw only if found ----------
        if found and final is not None:
            drawing.draw_tracking(
                frame,
                final,
                draw_color,
                label=label,
                pts=draw_pts,
                show_kp=show_kp,
            )

        if writer is not None:
            writer.write(frame)

        prev_gray = gray

        # ---------- 7. Throttling ----------
        if throttle:
            target = start_time + frame_idx / fps
            now = time.time()
            delay_ms = max(1, int((target - now) * 1000)) if now < target else 1
        else:
            delay_ms = 1

        cv2.imshow(config.WINDOW_NAME, frame)

        key = cv2.waitKey(delay_ms) & 0xFF
        if key == ord("q") or key == 27:
            break

        if not drawing.window_is_open():
            break

    cap.release()

    if writer is not None:
        writer.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()