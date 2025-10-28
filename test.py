import cv2

video_path = r"C:\Users\Nivedha\EVD-benchmark\videos\sample_video.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Cannot open video")
else:
    print("✅ Video opened successfully")
    ret, frame = cap.read()
    if ret:
        print("Frame shape:", frame.shape)
    cap.release()
