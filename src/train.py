"""
드론 탐지 모델 학습 스크립트 (1단계)
=====================================
- 모델: YOLOv8m (pretrained on COCO)
- 데이터셋: DUT Anti-UAV
- GPU: RTX 3070 Laptop (8GB VRAM)

사용법:
    python src/train.py
    python src/train.py --epochs 50 --batch 4  # VRAM 부족 시
"""

import argparse
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="드론 탐지 YOLOv8 학습")
    parser.add_argument("--model", type=str, default="E:/drone_runs/train/drone_det_v1/weights/best.pt",
                        help="모델: yolov8n/s/m/l/x.pt")
    parser.add_argument("--data", type=str, default="configs/dataset.yaml",
                        help="데이터셋 설정 파일")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8,
                        help="OOM 시 4로 줄이기")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--name", type=str, default="drone_det_v1")
    parser.add_argument("--resume", action="store_true",
                        help="이전 학습 이어하기")
    return parser.parse_args()


def train(args):
    print(f"\n{'='*60}")
    print(f"  모델: {args.model}")
    print(f"  데이터: {args.data}")
    print(f"  에폭: {args.epochs} | 배치: {args.batch} | 이미지: {args.imgsz}")
    print(f"{'='*60}\n")

    if args.resume:
        model = YOLO(f"E:/drone_runs/train/{args.name}/weights/last.pt")
        # results = model.train(resume=True) #이어서하기
    else:
        model = YOLO(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,

        # 옵티마이저
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=3,
        weight_decay=0.0005,

        # 드론 탐지 특화 augmentation
        hsv_h=0.015,        # 색상 변화 (야외 조명)
        hsv_s=0.7,          # 채도 변화
        hsv_v=0.4,          # 밝기 변화 (역광/그림자)
        degrees=15.0,        # 회전
        translate=0.1,
        scale=0.5,           # 스케일 변화 (거리별 크기, 핵심!)
        shear=2.0,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,          # 다양한 배경 합성
        mixup=0.1,
        copy_paste=0.1,      # 2025 챌린지 우승팀 핵심 기법

        # 학습 설정
        patience=15,         # Early stopping
        project="E:/drone_runs/train",
        name=args.name,
        exist_ok=True,
        seed=42,
        workers=4,
        device=0,
        amp=True,            # Mixed Precision (VRAM 절약)
        plots=True,          # 학습 곡선, confusion matrix
        save=True,
        save_period=10,
    )
    return results


def validate_best(args):
    """best 모델 최종 평가"""
    best_model = YOLO(f"runs/train/{args.name}/weights/best.pt")

    metrics = best_model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        split="test",
        plots=True,
        save_json=True,
    )

    print(f"\n{'='*60}")
    print(f"  최종 평가 결과 (best.pt)")
    print(f"{'='*60}")
    print(f"  mAP@50:    {metrics.box.map50:.4f}")
    print(f"  mAP@50-95: {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall:    {metrics.box.mr:.4f}")
    print(f"{'='*60}\n")
    return metrics


if __name__ == "__main__":
    args = parse_args()
    results = train(args)

    # 실제 저장된 경로에서 best.pt 찾기
    save_dir = results.save_dir
    best_path = f"{save_dir}/weights/best.pt"
    print(f"\n  모델 저장 위치: {best_path}")

    best_model = YOLO(best_path)
    metrics = best_model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        plots=True,
    )

    print(f"\n{'='*60}")
    print(f"  최종 평가 결과")
    print(f"{'='*60}")
    print(f"  mAP@50:    {metrics.box.map50:.4f}")
    print(f"  mAP@50-95: {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall:    {metrics.box.mr:.4f}")
    print(f"{'='*60}")