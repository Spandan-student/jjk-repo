import cv2
import mediapipe as mp
import csv
import os

# MediaPipe Hands Setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

DATASET_FILE = "hand_landmarks_dataset.csv"

def init_csv(filename):
    if not os.path.exists(filename):
        with open(filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            # 21 landmarks, each has x, y, z -> 63 features + 1 label
            header = ['label'] + [f'lm_{i}_{axis}' for i in range(21) for axis in ('x', 'y', 'z')]
            writer.writerow(header)

def main():
    label_name = input("Enter the label name for the hand sign you are about to record (e.g., ThumbsUp, Peace): ")
    
    init_csv(DATASET_FILE)
    
    cap = cv2.VideoCapture(0)
    
    print(f"\nRecording data for label: '{label_name}'")
    print("-> Press 's' to save the current frame's landmarks.")
    print("-> Press 'q' to quit collecting.")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        # Flip frame horizontally for natural viewing
        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        
        cv2.putText(frame, f"Label: {label_name} | Press 's' to save, 'q' to quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):
                    # Extract landmark coordinates
                    row = [label_name]
                    for lm in hand_landmarks.landmark:
                        row.extend([lm.x, lm.y, lm.z])
                        
                    # Append to CSV
                    with open(DATASET_FILE, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(row)
                    print(f"Saved 1 sample for '{label_name}'")
                    
        cv2.imshow('Data Collector', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
