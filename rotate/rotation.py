import cv2
import numpy as np
import argparse
from tqdm import tqdm
from sensation.segmentation import Segmentator

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)  # Get frames per second of the input video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames, fps


def segment_sidewalk(frame, model, original_size):
    # Resize frame to the size expected by the segmentation model
    resized_frame = cv2.resize(frame, (800, 640))
    frames = np.array([resized_frame])
    
    # Segmentator returns batch in this case batch is 1
    masks = model.inference(frames)

    # Resize mask back to the original frame size
    mask = (masks[0] == 2).astype(np.uint8) * 255
    resized_mask = cv2.resize(mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
    
    return resized_mask


def detect_keypoints(frame, mask):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = cv2.SIFT_create().detectAndCompute(gray, mask)
    return keypoints, descriptors

def visualize_keypoints(frame, keypoints):
    for kp in keypoints:
        x, y = np.int0(kp.pt)
        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

def annotate_frame_with_angle(frame, angle):
    annotated_frame = frame.copy()
    cv2.rectangle(annotated_frame, (10, 10), (400, 100), (255, 255, 255), -1)
    cv2.putText(annotated_frame, f"Rotation Angle: {angle:.2f}", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    return annotated_frame


def track_keypoints(frames, keypoints_descriptors, interval=60):
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    angles = []
    total_frames = len(frames)
    last_valid_angle = 0  # Initialize the angle to 0
    reference_image = 0
    
    for i in tqdm(range(0, total_frames - interval, interval), desc="Processing Frames"):
        old_frame = frames[reference_image]
        new_frame = frames[i + interval]

        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        new_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)

        p0 = np.array([kp.pt for kp in keypoints_descriptors[reference_image][0]], dtype=np.float32).reshape(-1, 1, 2)
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, p0, None, **lk_params)
        
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # Check if we have enough points
        if len(good_old) < 3 or len(good_new) < 3:
            print(f"Skipping frame {i} due to insufficient keypoints")
            angles.append(last_valid_angle)
        else:
            try:
                matrix, inliers = cv2.estimateAffinePartial2D(good_old, good_new)
                if matrix is None:
                    print(f"Skipping frame {i} due to transformation estimation failure")
                    angles.append(last_valid_angle)
                else:
                    angle = -np.arctan2(matrix[0, 1], matrix[0, 0]) * 180 / np.pi
                    last_valid_angle = angle
                    angles.append(angle)
            except cv2.error as e:
                print(f"Skipping frame {i} due to OpenCV error: {e}")
                angles.append(last_valid_angle)
        
        # Annotate only the new frame with the last known angle
        first_frame = i
        last_frame = i + interval
        for j in range(first_frame, last_frame):
            frames[j] = annotate_frame_with_angle(frames[j], last_valid_angle)
            visualize_keypoints(frames[j], keypoints_descriptors[j][0])
    
    return frames, angles


def main(input_video, output_video, model_path):
    # Load your segmentation model here
    model = Segmentator(model_path=model_path,
                        input_height=640,
                        input_width=800, csv_color_path="class_colors.csv")

    frames, fps = extract_frames(input_video)
    original_size = frames[0].shape[:2]
    segmented_masks = [segment_sidewalk(frame, model, original_size) for frame in tqdm(frames, desc="Segmenting Frames")]
    keypoints_descriptors = [detect_keypoints(frames[i], segmented_masks[i]) for i in range(len(frames))]
    annotated_frames, angles = track_keypoints(frames, keypoints_descriptors)
    
    # Write the annotated frames to the output video
    height, width, layers = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for frame in tqdm(annotated_frames, desc="Writing Frames"):
        out.write(frame)
    out.release()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process video for camera rotation measurement.")
    parser.add_argument("--input", required=True, help="Path to the input video")
    parser.add_argument("--output", required=True, help="Path to the output video")
    parser.add_argument("--model", required=True, help="Path to the segmentation model")

    args = parser.parse_args()
    main(args.input, args.output, args.model)
