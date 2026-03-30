"""
SAHI 기반 드론 탐지 추론 스크립트
==================================
왜 SAHI가 필요한가?
- YOLOv8 입력은 640x640 고정
- 1920x1080 이미지를 640으로 축소하면 소형 드론이 사라짐
- SAHI: 이미지를 640x640 패치로 잘라서 각각 추론 → mAP 5-12% 향상

사용법:
    python src/sahi_infer.py --source data/test_images/
    python src/sahi_infer.py --source data/test_images/ --compare
"""

import argparse
from pathlib import Path

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction, get_prediction


def parse_args():
    parser = argparse.ArgumentParser(description="SAHI 드론 탐지 추론")
    parser.add_argument("--model", type=str,
                        default="E:/drone_runs/train/drone_det_v2/weights/best.pt",
                        help="학습된 모델 경로")
    parser.add_argument("--source", type=str, required=True,
                        help="입력 이미지 경로 또는 폴더")
    parser.add_argument("--slice-size", type=int, default=640,
                        help="SAHI 슬라이스 크기")
    parser.add_argument("--overlap", type=float, default=0.25,
                        help="슬라이스 간 오버랩 비율 (0.2-0.3 권장)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="신뢰도 임계값")
    parser.add_argument("--compare", action="store_true",
                        help="SAHI vs 일반 추론 비교")
    parser.add_argument("--save-dir", type=str, default="runs/sahi_results")
    return parser.parse_args()


def load_model(model_path: str, conf: float = 0.25):
    """SAHI용 모델 로드"""
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=model_path,
        confidence_threshold=conf,
        device="cuda:0",
    )
    print(f"모델 로드 완료: {model_path}")
    return detection_model


def run_sahi_prediction(detection_model, image_path, slice_size=640, overlap=0.25):
    """
    SAHI 슬라이스 추론

    핵심 파라미터:
    - slice_size: 각 패치 크기 (모델 입력과 동일하게 640 권장)
    - overlap: 패치 간 겹침 (0.25 = 25%)
      → 겹치는 영역에서 탐지 누락 방지
      → 높을수록 정확하지만 느림
    """
    result = get_sliced_prediction(
        image=image_path,
        detection_model=detection_model,
        slice_height=slice_size,
        slice_width=slice_size,
        overlap_height_ratio=overlap,
        overlap_width_ratio=overlap,
        perform_standard_pred=True,   # 원본 전체 추론도 함께
        postprocess_type="NMS",       # 중복 박스 제거
        postprocess_match_threshold=0.5,
        verbose=1,
    )
    return result


def run_standard_prediction(detection_model, image_path):
    """일반 추론 (SAHI 미적용) - 비교용"""
    result = get_prediction(
        image=image_path,
        detection_model=detection_model,
        verbose=1,
    )
    return result


def compare_results(sahi_result, std_result, image_path):
    """SAHI vs 일반 추론 비교 출력"""
    sahi_count = len(sahi_result.object_prediction_list)
    std_count = len(std_result.object_prediction_list)

    print(f"\n{'='*60}")
    print(f"  SAHI vs 일반 추론 비교")
    print(f"  이미지: {Path(image_path).name}")
    print(f"{'='*60}")
    print(f"  일반 추론 탐지 수: {std_count}")
    print(f"  SAHI 추론 탐지 수: {sahi_count}")
    print(f"  추가 탐지:         +{max(0, sahi_count - std_count)}")
    print(f"{'='*60}")

    if sahi_count > 0:
        print("\n  [SAHI 탐지 상세]")
        for i, pred in enumerate(sahi_result.object_prediction_list):
            bbox = pred.bbox
            print(f"    #{i+1}: conf={pred.score.value:.3f} "
                  f"bbox=[{bbox.minx:.0f},{bbox.miny:.0f},"
                  f"{bbox.maxx:.0f},{bbox.maxy:.0f}] "
                  f"size={bbox.maxx-bbox.minx:.0f}x"
                  f"{bbox.maxy-bbox.miny:.0f}px")


def process_images(args):
    """이미지 처리 메인"""
    model = load_model(args.model, args.conf)
    source = Path(args.source)

    if source.is_file():
        image_paths = [source]
    elif source.is_dir():
        image_paths = sorted(
            list(source.glob("*.jpg")) + list(source.glob("*.png"))
        )
    else:
        print(f"소스를 찾을 수 없음: {args.source}")
        return

    print(f"\n처리할 이미지: {len(image_paths)}장\n")
    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    for img_path in image_paths:
        print(f"{'─'*40}")
        print(f"처리 중: {img_path.name}")

        # SAHI 추론
        sahi_result = run_sahi_prediction(
            model, str(img_path), args.slice_size, args.overlap
        )

        if args.compare:
            std_result = run_standard_prediction(model, str(img_path))
            compare_results(sahi_result, std_result, str(img_path))

        # 결과 시각화 저장
        sahi_result.export_visuals(
            export_dir=str(save_path),
            file_name=img_path.stem,
        )

    print(f"\n완료! 결과: {save_path}/")


if __name__ == "__main__":
    args = parse_args()
    process_images(args)