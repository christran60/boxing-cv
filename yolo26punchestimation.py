import cv2
import math
from ultralytics import YOLO

# Load the model
model = YOLO("yolo26n-pose.pt")

# Setup both webcams (Change indices 0 and 1 if iVCam or your main cam are on different IDs)
cap_front = cv2.VideoCapture(0)
cap_side = cv2.VideoCapture(1)

# Setup a resizable window that maintains its aspect ratio
window_name = "Multi-View Boxing Tracker"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

# Tracking Variables
punch_count = 0
l_punching = False
r_punching = False
# Add these with your tracking variables
l_cooldown = 0
r_cooldown = 0
COOLDOWN_FRAMES = 10  # How many frames to wait after a punch retracts (adjust based on your camera FPS)
# Thresholds
# Since your side camera looks further back in the room, you may need a slightly lower threshold for it.
FRONT_PUNCH_THRESH = 180  
FRONT_RETRACT_THRESH = 100    
SIDE_PUNCH_THRESH = 140  
SIDE_RETRACT_THRESH = 70

def process_pose_and_draw(frame, is_front_cam=True):
    """Runs YOLO, draws skeleton, and returns arm distances and head status."""
    results = list(model(frame, stream=True, verbose=False))
    l_dist, r_dist = 0, 0
    is_facing_forward = False

    for r in results:
        if r.keypoints is not None and len(r.keypoints.xy) > 0:
            kpts = r.keypoints.xy[0]

            if len(kpts) > 10:
                # --- JOINT EXTRACTION ---
                nose, l_eye, r_eye = kpts[0], kpts[1], kpts[2]
                ls, le, lw = kpts[5], kpts[7], kpts[9]
                rs, re, rw = kpts[6], kpts[8], kpts[10]

                # --- 1. DRAW ARMS SKELETON ---
                def draw_bone(pt1, pt2, color):
                    x1, y1 = int(pt1[0]), int(pt1[1])
                    x2, y2 = int(pt2[0]), int(pt2[1])
                    if (x1, y1) != (0, 0) and (x2, y2) != (0, 0):
                        cv2.line(frame, (x1, y1), (x2, y2), color, 4)
                        cv2.circle(frame, (x1, y1), 6, (0, 255, 0), -1)
                        cv2.circle(frame, (x2, y2), 6, (0, 255, 0), -1)

                draw_bone(ls, le, (255, 0, 0)) # Left Arm Blue
                draw_bone(le, lw, (255, 0, 0))
                draw_bone(rs, re, (0, 0, 255)) # Right Arm Red
                draw_bone(re, rw, (0, 0, 255))

                # --- 2. HEAD ORIENTATION (Only really matters for front cam) ---
                if is_front_cam and nose[0] != 0 and l_eye[0] != 0 and r_eye[0] != 0:
                    face_width = abs(l_eye[0] - r_eye[0])
                    if face_width > 0:
                        eyes_center_x = (l_eye[0] + r_eye[0]) / 2
                        nose_offset_pct = abs(nose[0] - eyes_center_x) / face_width
                        if nose_offset_pct < 0.25:
                            is_facing_forward = True

                # --- 3. ARM DISTANCES ---
                l_dist = math.hypot(lw[0] - ls[0], lw[1] - ls[1]) if lw[0] != 0 else 0
                r_dist = math.hypot(rw[0] - rs[0], rw[1] - rs[1]) if rw[0] != 0 else 0

    return frame, l_dist, r_dist, is_facing_forward

while cap_front.isOpened() and cap_side.isOpened():
    success_f, frame_f = cap_front.read()
    success_s, frame_s = cap_side.read()
    
    if not success_f or not success_s:
        print("Camera feed lost.")
        break

    # Process both frames
    frame_f, f_l_dist, f_r_dist, is_facing_forward = process_pose_and_draw(frame_f, is_front_cam=True)
    frame_s, s_l_dist, s_r_dist, _ = process_pose_and_draw(frame_s, is_front_cam=False)

# --- PUNCH COUNTING LOGIC (Anti-Double Count v2) ---
    
    # Decrease cooldown timers every frame
    if l_cooldown > 0: l_cooldown -= 1
    if r_cooldown > 0: r_cooldown -= 1

    # ================= LEFT ARM =================
    # 1. Check for Extension (Does either camera see a punch?)
    if f_l_dist > FRONT_PUNCH_THRESH or s_l_dist > SIDE_PUNCH_THRESH:
        if not l_punching and l_cooldown == 0:
            punch_count += 1
            l_punching = True
            
    # 2. Check for Valid Retraction
    # We check if either camera currently sees the arm hovering in the "extended" zone
    f_l_extended = f_l_dist >= FRONT_RETRACT_THRESH
    s_l_extended = s_l_dist >= SIDE_RETRACT_THRESH
    
    # Retract ONLY IF neither camera sees an extension AND at least one camera sees the arm (>0)
    elif not f_l_extended and not s_l_extended and (f_l_dist > 0 or s_l_dist > 0):
        if l_punching:
            l_punching = False
            l_cooldown = COOLDOWN_FRAMES

    # ================= RIGHT ARM =================
    # 1. Check for Extension
    if f_r_dist > FRONT_PUNCH_THRESH or s_r_dist > SIDE_PUNCH_THRESH:
        if not r_punching and r_cooldown == 0:
            punch_count += 1
            r_punching = True
            
    # 2. Check for Valid Retraction
    f_r_extended = f_r_dist >= FRONT_RETRACT_THRESH
    s_r_extended = s_r_dist >= SIDE_RETRACT_THRESH
    
    elif not f_r_extended and not s_r_extended and (f_r_dist > 0 or s_r_dist > 0):
        if r_punching:
            r_punching = False
            r_cooldown = COOLDOWN_FRAMES

    # --- UI & STITCHING ---
    # Resize frames to identical heights so they can be stitched horizontally
    target_height = 480
    h_f, w_f = frame_f.shape[:2]
    h_s, w_s = frame_s.shape[:2]
    
    frame_f_resized = cv2.resize(frame_f, (int(w_f * (target_height / h_f)), target_height))
    frame_s_resized = cv2.resize(frame_s, (int(w_s * (target_height / h_s)), target_height))

    # Stitch them together
    combined_frame = cv2.hconcat([frame_f_resized, frame_s_resized])

    # Draw Global HUD on the combined frame
    cv2.putText(combined_frame, f"Total Punches: {punch_count}", (40, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    
    # Debugging distances to help you calibrate the new side camera
    cv2.putText(combined_frame, f"Front L/R: {int(f_l_dist)}/{int(f_r_dist)}", (40, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(combined_frame, f"Side L/R: {int(s_l_dist)}/{int(s_r_dist)}", (40, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    if l_punching or r_punching:
        cv2.putText(combined_frame, "PUNCH ACTIVE", (40, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    status_text = "FORWARD" if is_facing_forward else "TURNED"
    color = (0, 255, 0) if is_facing_forward else (0, 165, 255)
    cv2.putText(combined_frame, f"Head: {status_text}", (40, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Render Window
    cv2.imshow(window_name, combined_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_front.release()
cap_side.release()
cv2.destroyAllWindows()