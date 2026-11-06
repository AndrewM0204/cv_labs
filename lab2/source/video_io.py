import cv2


def create_video_writer(path, fps, size, fourcc=None):
    if not path:
        return None

    if fps is None or not (fps > 0):
        fps = 30.0

    try:
        if fourcc and len(fourcc) == 4:
            code = cv2.VideoWriter_fourcc(*fourcc)
        else:
            ext = path.lower().rsplit(".", 1)[-1]

            if ext == "avi":
                code = cv2.VideoWriter_fourcc(*"MJPG")
            elif ext in ("mp4", "m4v") or ext == "mkv":
                code = cv2.VideoWriter_fourcc(*"mp4v")
            else:
                code = cv2.VideoWriter_fourcc(*"mp4v")

        writer = cv2.VideoWriter(path, code, fps, size)

        if writer.isOpened():
            return writer

        writer.release()
    except cv2.error:
        return None

    return None