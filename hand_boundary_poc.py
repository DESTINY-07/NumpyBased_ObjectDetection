import cv2
import numpy as np
import time
import math

CAM_ID = 0
FRAME_WIDTH = 640
MIN_CONTOUR_AREA = 3000

RECT_W_FRAC = 0.30
RECT_H_FRAC = 0.30

WARNING_DIST_FRAC = 0.12
DANGER_DIST_FRAC  = 0.05

SKIN_LOWER = np.array([0, 30, 60], dtype=np.uint8)
SKIN_UPPER = np.array([25, 255, 255], dtype=np.uint8)

KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

FONT = cv2.FONT_HERSHEY_SIMPLEX

def largest_contour(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < MIN_CONTOUR_AREA:
        return None
    return cnt

def fingertip_from_contour(cnt):
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    centroid = np.array([cx, cy])
    hull = cv2.convexHull(cnt, returnPoints=True)
    hull_pts = hull.reshape(-1,2)
    dists = np.linalg.norm(hull_pts - centroid[np.newaxis,:], axis=1)
    idx = np.argmax(dists)
    fingertip = tuple(hull_pts[idx].astype(int))
    return fingertip, (cx,cy)

def point_to_rect_distance(pt, rect):
    x, y = pt
    x1, y1, x2, y2 = rect
    dx = max(x1 - x, 0, x - x2)
    dy = max(y1 - y, 0, y - y2)
    return math.hypot(dx, dy)

def main():
    global WARNING_DIST_FRAC, DANGER_DIST_FRAC
    cap = cv2.VideoCapture(CAM_ID)
    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    fps_smooth = 0
    last_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        scale = FRAME_WIDTH / float(w)
        frame = cv2.resize(frame, (FRAME_WIDTH, int(h * scale)))
        H, W = frame.shape[:2]
        diag = math.hypot(W, H)
        rw = int(W * RECT_W_FRAC)
        rh = int(H * RECT_H_FRAC)
        cx, cy = W // 2, H // 2
        rect = (cx - rw//2, cy - rh//2, cx + rw//2, cy + rh//2)
        x1,y1,x2,y2 = rect
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, SKIN_LOWER, SKIN_UPPER)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL, iterations=1)
        mask = cv2.GaussianBlur(mask, (7,7), 0)
        cnt = largest_contour(mask)
        fingertip = None
        centroid = None
        if cnt is not None:
            cv2.drawContours(frame, [cnt], -1, (0,150,0), 2)
            hull = cv2.convexHull(cnt)
            cv2.drawContours(frame, [hull], -1, (0,255,255), 2)
            res = fingertip_from_contour(cnt)
            if res is not None:
                fingertip, centroid = res
                cv2.circle(frame, fingertip, 8, (0,0,255), -1)
                cv2.circle(frame, centroid, 5, (255,0,0), -1)
                cv2.putText(frame, f"Area:{int(cv2.contourArea(cnt))}", (10, H-10), FONT, 0.5, (255,255,255), 1)
        state = "SAFE"
        if fingertip is not None:
            dist_px = point_to_rect_distance(fingertip, rect)
            norm = dist_px / diag
            if dist_px == 0:
                state = "DANGER"
            elif norm <= DANGER_DIST_FRAC:
                state = "DANGER"
            elif norm <= WARNING_DIST_FRAC:
                state = "WARNING"
            else:
                state = "SAFE"
            cv2.putText(frame, f"dist_px:{int(dist_px)} norm:{norm:.3f}", (10, H-30), FONT, 0.5, (200,200,200), 1)
        else:
            cv2.putText(frame, "No hand detected", (10, H-30), FONT, 0.6, (0,0,255), 1)
        color = (0,255,0)
        if state == "WARNING":
            color = (0,165,255)
        elif state == "DANGER":
            color = (0,0,255)
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 3)
        cv2.putText(frame, f"STATE: {state}", (10,30), FONT, 1.0, color, 2)
        if state == "DANGER":
            cv2.putText(frame, "DANGER DANGER", (W//6, H//2), FONT, 2.0, (0,0,255), 4)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        small_mask = cv2.resize(mask_bgr, (int(W*0.3), int(H*0.3)))
        mx, my = W - small_mask.shape[1] - 10, 10
        frame[my:my+small_mask.shape[0], mx:mx+small_mask.shape[1]] = small_mask
        now = time.time()
        dt = now - last_time
        last_time = now
        fps = 1.0 / dt if dt > 0 else 0.0
        fps_smooth = 0.85 * fps_smooth + 0.15 * fps
        cv2.putText(frame, f"FPS: {fps_smooth:.1f}", (W-140, H-10), FONT, 0.6, (255,255,0), 2)
        cv2.imshow("Hand Boundary POC", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break
        if key == ord('u'):
            WARNING_DIST_FRAC *= 1.1
            DANGER_DIST_FRAC *= 1.1
            print("Increased thresholds")
        if key == ord('j'):
            WARNING_DIST_FRAC *= 0.9
            DANGER_DIST_FRAC *= 0.9
            print("Decreased thresholds")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
