# Hand Boundary Detection

This project is a proof-of-concept for a real-time hand boundary detection system using OpenCV and Python. It captures video from a webcam, identifies the user's hand, and tracks its boundary. The primary goal is to demonstrate how to achieve hand detection and proximity alerts using fundamental computer vision techniques with OpenCV and NumPy, without relying on specialized libraries like MediaPipe or OpenPose.

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/CV2_object_detection.git
    cd CV2_object_detection
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**
    -   On Windows:
        ```bash
        venv\Scripts\activate
        ```
    -   On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```

4.  **Install the dependencies:**
    ```bash
    pip install -r req.txt
    ```

5.  **Run the script:**
    ```bash
    python hand_boundary_poc.py
    ```

## Expected Results

When you run the script, you will see a window displaying your webcam feed. The application will:
-   Detect your hand when it's visible in the frame.
-   Draw a green contour around your hand.
-   Draw a yellow convex hull around the hand contour.
-   Identify and mark the fingertip with a red circle.
-   Display a central rectangle that acts as a "safe zone."
-   Show the state as "SAFE", "WARNING", or "DANGER" based on the proximity of your fingertip to the central rectangle.
-   The rectangle's border will change color from green (SAFE) to orange (WARNING) to red (DANGER).

## Real-world Applications

This type of hand tracking and proximity detection can be used in various applications, such as:

-   **Gesture Control:** Control applications and devices using hand gestures.
-   **Virtual Reality (VR) and Augmented Reality (AR):** Interact with virtual objects in a VR/AR environment.
-   **Human-Computer Interaction (HCI):** Create more intuitive user interfaces.
-   **Safety Systems:**  In industrial settings, this could be used to detect if a worker's hand is too close to a dangerous machine and trigger an alert.

## Code Snippets

The core logic of this project is based on a few key image processing steps.

### Skin Detection

The first step is to isolate the hand from the background. This is done by converting the image to the HSV color space and then applying a color mask to detect skin tones.

```python
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, SKIN_LOWER, SKIN_UPPER)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL, iterations=2)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL, iterations=1)
mask = cv2.GaussianBlur(mask, (7,7), 0)
```

### Finding the Largest Contour

Once we have the skin mask, we find all the contours and select the largest one, which is assumed to be the hand.

```python
def largest_contour(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < MIN_CONTOUR_AREA:
        return None
    return cnt
```

### Fingertip Detection

The fingertip is found by calculating the convex hull of the hand contour and then finding the point on the hull that is farthest from the contour's centroid.

```python
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
```

## Dependencies

-   numpy
-   opencv-python
