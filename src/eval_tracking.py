"""
Phase 2: 추적 정확도 평가
- GT(정답)와 예측 결과를 비교하여 MOTA, IDF1 등 추적 지표 계산
- motmetrics 라이브러리 사용
"""

import motmetrics as mm
import numpy as np

np.asfarray = lambda x, dtype=np.float64: np.asarray(x, dtype=dtype)

# ============ 설정 ============
VIDEO_NAME = "video01"
GT_PATH = f"data/Anti-UAV-Tracking-V0GT/{VIDEO_NAME}_gt.txt"
PRED_PATH = f"results/tracking_{VIDEO_NAME}_pred.txt"

# ============ GT 로드 ============
print(f"GT 로드: {GT_PATH}")
gt_data = {}
with open(GT_PATH, "r") as f:
    for i, line in enumerate(f):
        parts = line.strip().split()
        if len(parts) == 4:
            x, y, w, h = map(int, parts)
            # GT는 드론 1대이므로 ID=0 고정
            gt_data[i + 1] = [(0, x, y, w, h)]

print(f"  GT 프레임 수: {len(gt_data)}")

# ============ 예측 로드 ============
print(f"예측 로드: {PRED_PATH}")
pred_data = {}
with open(PRED_PATH, "r") as f:
    for line in f:
        parts = line.strip().split(",")
        frame = int(parts[0])
        track_id = int(parts[1])
        x, y, w, h = int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])
        if frame not in pred_data:
            pred_data[frame] = []
        pred_data[frame].append((track_id, x, y, w, h))

print(f"  예측 프레임 수: {len(pred_data)}")

# ============ motmetrics로 평가 ============
print("\n평가 중...")
acc = mm.MOTAccumulator(auto_id=True)

# 전체 프레임 순회
total_frames = max(max(gt_data.keys()), max(pred_data.keys()))

for frame_id in range(1, total_frames + 1):
    # GT 박스
    gt_boxes = gt_data.get(frame_id, [])
    gt_ids = [item[0] for item in gt_boxes]
    gt_rects = [[item[1], item[2], item[3], item[4]] for item in gt_boxes]

    # 예측 박스
    pred_boxes = pred_data.get(frame_id, [])
    pred_ids = [item[0] for item in pred_boxes]
    pred_rects = [[item[1], item[2], item[3], item[4]] for item in pred_boxes]

    # IoU 거리 행렬 계산
    if len(gt_rects) > 0 and len(pred_rects) > 0:
        distances = mm.distances.iou_matrix(
            np.array(gt_rects),
            np.array(pred_rects),
            max_iou=0.5  # IoU 0.5 이하는 매칭 안 함
        )
    else:
        distances = np.empty((len(gt_rects), len(pred_rects)))

    acc.update(gt_ids, pred_ids, distances)

# ============ 결과 출력 ============
mh = mm.metrics.create()
summary = mh.compute(acc, metrics=[
    "num_frames",     # 총 프레임 수
    "mota",           # 추적 정확도 (종합)
    "motp",           # 추적 정밀도 (위치)
    "idf1",           # ID 일관성
    "num_switches",   # ID 전환 횟수
    "num_misses",     # 미탐지 횟수
    "num_false_positives",  # 오탐 횟수
    "mostly_tracked",      # 대부분 추적 성공한 객체 수
    "mostly_lost",         # 대부분 놓친 객체 수
], name=VIDEO_NAME)

print("\n" + "=" * 60)
print("추적 평가 결과")
print("=" * 60)

# 보기 좋게 출력
results = summary.to_dict("records")[0]
print(f"  MOTA  (추적 정확도):    {results['mota']:.1%}")
print(f"  MOTP  (위치 정밀도):    {results['motp']:.4f}")
print(f"  IDF1  (ID 일관성):      {results['idf1']:.1%}")
print(f"  ID Switches:            {results['num_switches']}")
print(f"  Misses (미탐지):        {results['num_misses']}")
print(f"  False Positives (오탐): {results['num_false_positives']}")
print(f"  Mostly Tracked:         {results['mostly_tracked']}")
print(f"  Mostly Lost:            {results['mostly_lost']}")
print(f"  총 프레임:              {results['num_frames']}")
print("=" * 60)