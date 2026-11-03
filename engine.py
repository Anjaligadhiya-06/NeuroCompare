# engine.py
import cv2
import numpy as np

def align_and_register(baseline_img, followup_img):
    """
    Aligns the follow-up image to match the baseline image using OpenCV's ORB feature matching.
    (A highly reliable, fast approach for a 48-hour hackathon hack).
    """
    # Convert to grayscale if color
    if len(baseline_img.shape) == 3:
        gray_base = cv2.cvtColor(baseline_img, cv2.COLOR_BGR2GRAY)
        gray_follow = cv2.cvtColor(followup_img, cv2.COLOR_BGR2GRAY)
    else:
        gray_base = baseline_img
        gray_follow = followup_img

    # 1. Detect features using ORB
    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(gray_base, None)
    kp2, des2 = orb.detectAndCompute(gray_follow, None)

    # 2. Match features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # 3. Extract spatial coordinates of matches
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # 4. Find the transformation matrix (Homography) and warp the follow-up image
    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    aligned_followup = cv2.warpPerspective(followup_img, M, (baseline_img.shape[1], baseline_img.shape[0]))
    
    return aligned_followup

def compute_progression(baseline_img, aligned_followup):
    """
    Compares two aligned images and highlights structural tissue growth or shrinkage.
    """
    # Absolute difference between images
    diff = cv2.absdiff(baseline_img, aligned_followup)
    
    # Convert difference to grayscale to create a single-channel mask
    if len(diff.shape) == 3:
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    else:
        diff_gray = diff

    # Threshold the differences to filter out minor camera noise
    _, thresh = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)
    
    # Generate a beautiful RGB visual overlay
    overlay = aligned_followup.copy()
    
    # FIX IS HERE: Only paint if there are actual pixels detected
    if np.any(thresh > 0):
        overlay[thresh > 0] = [0, 0, 255]  # Paint differences in bright red BGR
    
    # Calculate percentage change based on changed pixels vs total image space
    changed_pixels = np.sum(thresh > 0)
    total_pixels = thresh.size
    pct_change = (changed_pixels / total_pixels) * 100
    
    return overlay, pct_change
