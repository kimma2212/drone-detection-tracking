"""
Phase 2: ByteTrack Video Tracking
- 프레임 이미지들을 읽어서 YOLOv8 + ByteTrack으로 드론 추적
- 각 드론에 고유 ID를 부여하고 궤적을 표시
- 결과를 영상 파일(.mp4)로 저장
"""

import cv2
import os
import glob
from ultralytics import YOLO
from collections import defaultdict
import numpy as np

# ============ 설정 ============
MODEL_PATH = "E:/drone_runs/train/drone_det_v2/weights/best.pt"
VIDEO_DIR = "data/Anti-UAV-Tracking-V0/video01"
OUTPUT_PATH = "results/tracking_video01.mp4"
CONFIDENCE = 0.55      # 추적용이라 좀 낮게 (recall 우선)
FPS = 30

# ============ 모델 로드 ============
print("모델 로딩...")
model = YOLO(MODEL_PATH)
print("모델 로드 완료!")

# ============ 프레임 로드 ============
frame_files = sorted(glob.glob(os.path.join(VIDEO_DIR, "*.jpg")))
print(f"총 프레임 수: {len(frame_files)}")

if len(frame_files) == 0:
    print("프레임을 찾을 수 없습니다. 경로를 확인하세요.")
    exit()

# 첫 프레임으로 영상 크기 확인
first_frame = cv2.imread(frame_files[0])
h, w = first_frame.shape[:2]
print(f"해상도: {w}x{h}")

# ============ 영상 저장 준비 ============
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, FPS, (w, h))

# ============ 추적 실행 ============
# 각 ID별 궤적 저장 (궤적 시각화용)
track_history = defaultdict(list)

frame_predictions = {}




# 색상표 (ID별로 다른 색)
COLORS = [
    (0, 255, 0),    # 초록
    (255, 0, 0),    # 파랑
    (0, 0, 255),    # 빨강
    (255, 255, 0),  # 하늘
    (0, 255, 255),  # 노랑
    (255, 0, 255),  # 보라
]

print("추적 시작...")
for i, frame_file in enumerate(frame_files):
    frame = cv2.imread(frame_file)

    # model.track() = 탐지 + 추적 (ByteTrack 내장)
    results = model.track(
        frame,
        conf=CONFIDENCE,
        tracker="src/bytetrack_custom.yaml",  # ByteTrack 사용
        persist=True,              # 프레임 간 ID 유지
        verbose=False
    )

    # 결과 시각화
    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        confs = results[0].boxes.conf.cpu().numpy()

        for box, track_id, conf in zip(boxes, track_ids, confs):
            x1, y1, x2, y2 = box
            color = COLORS[track_id % len(COLORS)]

            # 바운딩 박스
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

            # ID + confidence 라벨
            label = f"Drone #{track_id} {conf:.0%}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # 궤적 저장 (박스 중심점)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            track_history[track_id].append((cx, cy))

            # 궤적 그리기 (최근 50 프레임)
            points = track_history[track_id][-50:]
            for j in range(1, len(points)):
                thickness = max(1, j // 5)  # 최근일수록 굵게
                cv2.line(frame, points[j-1], points[j], color, thickness)


            # 예측 결과 기록
            if i not in frame_predictions:
                frame_predictions[i] = []
            frame_predictions[i].append((track_id, x1, y1, x2-x1, y2-y1, conf))



    # 프레임 번호 표시
    cv2.putText(frame, f"Frame: {i+1}/{len(frame_files)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    writer.write(frame)

    # 진행 상황 (100프레임마다)
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{len(frame_files)} 프레임 처리 완료")

writer.release()

# ============ 예측 결과 저장 ============
pred_path = OUTPUT_PATH.replace(".mp4", "_pred.txt")
with open(pred_path, "w") as f:
    for frame_idx in range(len(frame_files)):
        if frame_idx in frame_predictions:
            for track_id, x, y, w, h, conf in frame_predictions[frame_idx]:
                f.write(f"{frame_idx+1},{track_id},{x},{y},{w},{h},{conf:.4f}\n")

print(f"\n완료! 결과 저장: {OUTPUT_PATH}")
print(f"예측 데이터 저장: {pred_path}")
print(f"총 감지된 고유 ID 수: {len(track_history)}")