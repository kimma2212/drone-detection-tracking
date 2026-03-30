"""
SAHI 적용 전후 mAP 비교 평가
==============================
Ground Truth와 대조하여 정확한 mAP를 측정

사용법:
    python configs/src/sahi_eval.py
    python configs/src/sahi_eval.py --max-images 100
"""

import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction, get_prediction
import cv2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="E:/drone_runs/train/drone_det_v2/weights/best.pt")
    parser.add_argument("--img-dir", default="data/dut-anti-uav/images/val")
    parser.add_argument("--label-dir", default="data/dut-anti-uav/labels/val")
    parser.add_argument("--max-images", type=int, default=None,
                        help="분석할 최대 이미지 수 (None이면 전체)")
    parser.add_argument("--conf", type=float, default=0.25)
    return parser.parse_args()


def load_gt(label_path, img_w, img_h):
    """YOLO txt 라벨 → 픽셀 좌표 변환"""
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().strip().split("\n"):
        if not line.strip():
            continue
        parts = line.strip().split()
        if len(parts) >= 5:
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = (cx - w/2) * img_w
            y1 = (cy - h/2) * img_h
            x2 = (cx + w/2) * img_w
            y2 = (cy + h/2) * img_h
            area = w * h
            if area < 0.001:
                size = "small"
            elif area < 0.01:
                size = "medium"
            else:
                size = "large"
            boxes.append({"bbox": [x1, y1, x2, y2], "size": size})
    return boxes


def compute_iou(box1, box2):
    """두 박스의 IoU 계산"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    
    return inter / max(union, 1e-6)


def match_predictions(preds, gts, iou_threshold=0.5):
    """예측과 GT를 매칭하여 TP/FP 판정"""
    if not preds or not gts:
        return [0] * len(preds), len(gts)
    
    # confidence 높은 순 정렬
    preds_sorted = sorted(preds, key=lambda x: x["conf"], reverse=True)
    gt_matched = [False] * len(gts)
    results = []  # 1=TP, 0=FP
    
    for pred in preds_sorted:
        best_iou = 0
        best_gt_idx = -1
        for j, gt in enumerate(gts):
            if gt_matched[j]:
                continue
            iou = compute_iou(pred["bbox"], gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j
        
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            results.append(1)  # TP
            gt_matched[best_gt_idx] = True
        else:
            results.append(0)  # FP
    
    fn = sum(1 for m in gt_matched if not m)  # 놓친 GT
    return results, fn


def compute_ap(tp_list, total_gt):
    """AP (Average Precision) 계산"""
    if total_gt == 0:
        return 0.0
    
    tp_cumsum = np.cumsum(tp_list)
    fp_cumsum = np.cumsum([1 - t for t in tp_list])
    
    recalls = tp_cumsum / total_gt
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    
    # AP 계산 (11-point interpolation)
    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        prec_at_recall = [p for p, r in zip(precisions, recalls) if r >= t]
        if prec_at_recall:
            ap += max(prec_at_recall) / 11
    return ap


def run_evaluation(model, image_paths, label_dir, use_sahi=False):
    """전체 이미지에 대해 평가 실행"""
    all_preds = []
    all_gts = []
    total_gt = 0
    size_gt = defaultdict(int)
    size_tp = defaultdict(int)
    
    for i, img_path in enumerate(image_paths):
        if (i + 1) % 100 == 0:
            print(f"    진행: {i+1}/{len(image_paths)}")
        
        # 이미지 크기
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]
        
        # GT 로드
        label_path = Path(label_dir) / (img_path.stem + ".txt")
        gts = load_gt(label_path, img_w, img_h)
        total_gt += len(gts)
        for gt in gts:
            size_gt[gt["size"]] += 1
        
        # 추론
        if use_sahi:
            result = get_sliced_prediction(
                image=str(img_path),
                detection_model=model,
                slice_height=640,
                slice_width=640,
                overlap_height_ratio=0.25,
                overlap_width_ratio=0.25,
                perform_standard_pred=True,
                postprocess_type="NMS",
                postprocess_match_threshold=0.5,
                verbose=0,
            )
        else:
            result = get_prediction(
                image=str(img_path),
                detection_model=model,
                verbose=0,
            )
        
        preds = []
        for pred in result.object_prediction_list:
            bbox = pred.bbox
            preds.append({
                "bbox": [bbox.minx, bbox.miny, bbox.maxx, bbox.maxy],
                "conf": pred.score.value,
            })
        
        # IoU 0.5에서 매칭
        tp_fp, fn = match_predictions(preds, gts, iou_threshold=0.5)
        
        # 크기별 TP 카운트
        gt_matched = [False] * len(gts)
        for pred in sorted(preds, key=lambda x: x["conf"], reverse=True):
            for j, gt in enumerate(gts):
                if gt_matched[j]:
                    continue
                iou = compute_iou(pred["bbox"], gt["bbox"])
                if iou >= 0.5:
                    gt_matched[j] = True
                    size_tp[gt["size"]] += 1
                    break
        
        all_preds.extend([(p["conf"], tp) for p, tp in zip(
            sorted(preds, key=lambda x: x["conf"], reverse=True), tp_fp)])
    
    # 전체 AP 계산
    all_preds.sort(key=lambda x: x[0], reverse=True)
    tp_list = [x[1] for x in all_preds]
    ap50 = compute_ap(tp_list, total_gt)
    
    total_tp = sum(tp_list)
    total_fp = len(tp_list) - total_tp
    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_gt, 1)
    
    return {
        "ap50": ap50,
        "precision": precision,
        "recall": recall,
        "total_gt": total_gt,
        "total_pred": len(all_preds),
        "total_tp": total_tp,
        "total_fp": total_fp,
        "size_gt": dict(size_gt),
        "size_tp": dict(size_tp),
    }


def print_results(name, results):
    """결과 출력"""
    print(f"\n  [{name}]")
    print(f"    mAP@50:    {results['ap50']:.4f}")
    print(f"    Precision: {results['precision']:.4f}")
    print(f"    Recall:    {results['recall']:.4f}")
    print(f"    탐지: {results['total_pred']}개 (TP:{results['total_tp']} FP:{results['total_fp']})")
    print(f"    GT:   {results['total_gt']}개")
    
    print(f"\n    [크기별 Recall]")
    for size in ["small", "medium", "large"]:
        gt = results["size_gt"].get(size, 0)
        tp = results["size_tp"].get(size, 0)
        r = tp / max(gt, 1)
        print(f"      {size:8s}: {tp}/{gt} = {r:.1%}")


def main():
    args = parse_args()
    
    model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=args.model,
        confidence_threshold=args.conf,
        device="cuda:0",
    )
    
    img_dir = Path(args.img_dir)
    image_paths = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    if args.max_images:
        image_paths = image_paths[:args.max_images]
    
    print(f"\n{'='*60}")
    print(f"  SAHI 적용 전후 mAP 비교")
    print(f"  이미지: {len(image_paths)}장 | conf: {args.conf}")
    print(f"{'='*60}")
    
    # 일반 추론
    print(f"\n  일반 추론 평가 중...")
    std_results = run_evaluation(model, image_paths, args.label_dir, use_sahi=False)
    print_results("일반 추론", std_results)
    
    # SAHI 추론
    print(f"\n  SAHI 추론 평가 중...")
    sahi_results = run_evaluation(model, image_paths, args.label_dir, use_sahi=True)
    print_results("SAHI 추론", sahi_results)
    
    # 비교
    print(f"\n{'='*60}")
    print(f"  비교 요약")
    print(f"{'='*60}")
    
    ap_diff = sahi_results["ap50"] - std_results["ap50"]
    p_diff = sahi_results["precision"] - std_results["precision"]
    r_diff = sahi_results["recall"] - std_results["recall"]
    
    print(f"    mAP@50:    {std_results['ap50']:.4f} → {sahi_results['ap50']:.4f} ({ap_diff:+.4f})")
    print(f"    Precision: {std_results['precision']:.4f} → {sahi_results['precision']:.4f} ({p_diff:+.4f})")
    print(f"    Recall:    {std_results['recall']:.4f} → {sahi_results['recall']:.4f} ({r_diff:+.4f})")
    
    print(f"\n    [크기별 Recall 비교]")
    for size in ["small", "medium", "large"]:
        gt = std_results["size_gt"].get(size, 0)
        std_tp = std_results["size_tp"].get(size, 0)
        sahi_tp = sahi_results["size_tp"].get(size, 0)
        std_r = std_tp / max(gt, 1)
        sahi_r = sahi_tp / max(gt, 1)
        diff = sahi_r - std_r
        print(f"      {size:8s}: {std_r:.1%} → {sahi_r:.1%} ({diff:+.1%})")
    
    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()