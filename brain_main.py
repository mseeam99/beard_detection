
#!/usr/bin/env python3
"""
beard_scanner.py  v2 — Real-time face scan + beard mapping
(beard-trimming helmet project, Phase 1 perception pipeline)

v2 changes:
  * Beard zone now EXTENDS BELOW THE JAW to cover the chin underside
    and neck beard (MediaPipe has no neck landmarks, so we extrude
    the jawline downward along the face's vertical axis).
  * More sensitive segmentation (catches stubble + lighter hair).
  * 'f' key: full-zone mode — treat the ENTIRE beard zone as beard,
    which is what the trimmer planner usually wants anyway.

Install (inside your beardenv):
    pip install opencv-python "mediapipe==0.10.14" numpy

Keys:
    q  quit
    m  toggle face mesh overlay
    b  toggle beard mask overlay
    f  toggle FULL-ZONE mode (whole zone = beard)
    z  toggle pseudo-depth coloring of landmarks
    s  save scan  ->  scan_beard_mask.png + scan_landmarks.npy
    [ / ]  decrease / increase detection sensitivity
    , / .  shrink / grow the neck extension below the jaw
"""

import time
import numpy as np
import cv2
import mediapipe as mp

# ---------------------------------------------------------------
# Optional RealSense depth support (auto-detected)
# ---------------------------------------------------------------
USE_REALSENSE = False
try:
    import pyrealsense2 as rs
    if len(rs.context().devices) > 0:
        USE_REALSENSE = True
except Exception:
    pass

# ---------------------------------------------------------------
# Beard-zone boundary landmark IDs (MediaPipe FaceMesh)
# ---------------------------------------------------------------
TOP_IDS = [234, 50, 2, 280, 454]                       # ear-cheek-nose-cheek-ear
JAW_IDS = [361, 288, 397, 365, 379, 378, 400, 377,     # right jaw -> chin
           152,                                        # chin tip
           148, 176, 149, 150, 136, 172, 58, 132, 93]  # chin -> left jaw

SKIN_REF_IDS = [10, 108, 337, 195]   # forehead x3 + nose bridge (bare skin)

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


def landmarks_to_px(landmarks, w, h):
    return np.array([[lm.x * w, lm.y * h, lm.z] for lm in landmarks],
                    dtype=np.float32)


def beard_zone_mask(pts, w, h, neck_frac):
    """
    Beard zone polygon. The jawline points are duplicated and pushed
    DOWNWARD (along the forehead->chin axis, so it works with head tilt)
    by neck_frac * face_height, so under-chin and neck beard is included.
    """
    face_axis = pts[152, :2] - pts[10, :2]              # forehead -> chin
    n = np.linalg.norm(face_axis)
    down = face_axis / n if n > 1e-3 else np.array([0.0, 1.0])
    shift = down * (neck_frac * n)

    top = pts[TOP_IDS, :2]
    jaw = pts[JAW_IDS, :2]
    jaw_low = jaw + shift                                # extruded neck edge

    # polygon: top boundary L->R, then down the right side along the
    # extruded jaw, back along it R->L... simplest closed shape:
    # top boundary, then the extruded jawline (which already runs
    # right ear -> chin -> left ear).
    poly = np.vstack([top, jaw_low]).astype(np.int32)

    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    return mask, poly


def skin_reference_lab(frame_lab, pts, w, h, r=12):
    samples = []
    for i in SKIN_REF_IDS:
        x, y = int(pts[i, 0]), int(pts[i, 1])
        patch = frame_lab[max(0, y-r):min(h, y+r),
                          max(0, x-r):min(w, x+r)].reshape(-1, 3)
        if len(patch):
            samples.append(patch)
    return np.median(np.vstack(samples), axis=0) if samples else None


def segment_beard(frame_bgr, zone_mask, skin_lab, sensitivity):
    """
    Beard = pixels in the zone that are darker / different from the
    skin reference OR show hair texture. v2: lower threshold, heavier
    texture weight, and a dilation pass so sparse stubble merges into
    contiguous regions (what a trimmer planner needs).
    """
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    d_l = skin_lab[0] - lab[:, :, 0]
    d_ab = np.sqrt((lab[:, :, 1] - skin_lab[1]) ** 2 +
                   (lab[:, :, 2] - skin_lab[2]) ** 2)
    color_score = np.clip(d_l, 0, None) * 1.6 + d_ab

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    texture = cv2.boxFilter(cv2.magnitude(gx, gy), -1, (9, 9))

    score = color_score + 0.55 * texture
    thresh = 26.0 * (2.0 - sensitivity)        # v2: lower base (was 38)
    beard = ((score > thresh).astype(np.uint8)) * 255
    beard = cv2.bitwise_and(beard, zone_mask)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    beard = cv2.morphologyEx(beard, cv2.MORPH_OPEN, k)
    beard = cv2.dilate(beard, k, iterations=1)          # merge stubble
    beard = cv2.morphologyEx(beard, cv2.MORPH_CLOSE, k, iterations=3)
    return cv2.bitwise_and(beard, zone_mask)


def draw_depth_landmarks(frame, pts):
    z = pts[:, 2]
    zn = (z - z.min()) / max(1e-6, (z.max() - z.min()))
    for (x, y, _), t in zip(pts, zn):
        cv2.circle(frame, (int(x), int(y)), 1,
                   (int(255 * (1 - t)), 60, int(255 * t)), -1)


def open_camera():
    if USE_REALSENSE:
        pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        profile = pipe.start(cfg)
        align = rs.align(rs.stream.color)
        scale = profile.get_device().first_depth_sensor().get_depth_scale()

        def get():
            f = align.process(pipe.wait_for_frames())
            c = np.asanyarray(f.get_color_frame().get_data())
            d = np.asanyarray(f.get_depth_frame().get_data())
            return c, d * scale * 1000.0
        return get, pipe.stop

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam (index 0).")

    def get():
        ok, frame = cap.read()
        if not ok:
            raise SystemExit("Webcam stream ended.")
        return frame, None
    return get, cap.release


def main():
    get_frames, close = open_camera()
    mode = "RealSense DEPTH" if USE_REALSENSE else "Webcam (2D)"
    print(f"[beard_scanner] camera mode: {mode}")

    show_mesh, show_mask, show_z = True, True, False
    full_zone = False
    sensitivity = 1.25          # v2: hotter default (was 1.0)
    neck_frac = 0.22            # neck extension = 22% of face height
    last_scan = None
    fps_t, fps = time.time(), 0.0

    with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                               min_detection_confidence=0.6,
                               min_tracking_confidence=0.6) as face_mesh:
        while True:
            frame, depth_mm = get_frames()
            frame = cv2.flip(frame, 1)
            if depth_mm is not None:
                depth_mm = cv2.flip(depth_mm, 1)
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            result = face_mesh.process(rgb)

            hud = [f"mode: {mode}" + ("   [FULL-ZONE]" if full_zone else ""),
                   f"sensitivity {sensitivity:.2f} ([ ])   "
                   f"neck extend {neck_frac:.2f} (, .)"]

            if result.multi_face_landmarks:
                lms = result.multi_face_landmarks[0]
                pts = landmarks_to_px(lms.landmark, w, h)
                zone, poly = beard_zone_mask(pts, w, h, neck_frac)

                lab_img = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
                skin = skin_reference_lab(lab_img, pts, w, h)

                if skin is not None:
                    if full_zone:
                        beard = zone.copy()
                    else:
                        beard = segment_beard(frame, zone, skin, sensitivity)
                    coverage = 100.0 * cv2.countNonZero(beard) / max(
                        1, cv2.countNonZero(zone))
                    hud.append(f"beard coverage of zone: {coverage:5.1f} %")

                    if show_mask:
                        ov = frame.copy()
                        ov[beard > 0] = (0, 80, 255)
                        frame = cv2.addWeighted(ov, 0.45, frame, 0.55, 0)

                    if depth_mm is not None:
                        d = depth_mm[beard > 0]
                        d = d[(d > 100) & (d < 800)]
                        if len(d):
                            hud.append(f"beard surface distance: "
                                       f"{np.median(d):.0f} mm (median)")

                    last_scan = (beard, pts)

                cv2.polylines(frame, [poly], True, (0, 255, 120), 2)

                if show_mesh:
                    mp_drawing.draw_landmarks(
                        frame, lms, mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_styles
                        .get_default_face_mesh_tesselation_style())
                if show_z:
                    draw_depth_landmarks(frame, pts)
            else:
                hud.append("no face detected")
                last_scan = None

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, now - fps_t))
            fps_t = now
            hud.append(f"{fps:4.1f} fps  q quit  m mesh  b mask  "
                       f"f full-zone  z depth  s save")
            for i, line in enumerate(hud):
                cv2.putText(frame, line, (12, 28 + 24 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4)
                cv2.putText(frame, line, (12, 28 + 24 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1)

            cv2.imshow("Beard Scanner v2", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                show_mesh = not show_mesh
            elif key == ord('b'):
                show_mask = not show_mask
            elif key == ord('f'):
                full_zone = not full_zone
            elif key == ord('z'):
                show_z = not show_z
            elif key == ord('['):
                sensitivity = max(0.0, sensitivity - 0.05)
            elif key == ord(']'):
                sensitivity = min(2.0, sensitivity + 0.05)
            elif key == ord(','):
                neck_frac = max(0.0, neck_frac - 0.02)
            elif key == ord('.'):
                neck_frac = min(0.5, neck_frac + 0.02)
            elif key == ord('s') and last_scan is not None:
                beard, pts = last_scan
                cv2.imwrite("scan_beard_mask.png", beard)
                np.save("scan_landmarks.npy", pts)
                print("[beard_scanner] saved scan_beard_mask.png "
                      "and scan_landmarks.npy")

    close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()




