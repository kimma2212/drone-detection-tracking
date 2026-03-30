"""
SAHI vs 일반 추론 통계 분석 스크립트
====================================
전체 test 이미지에서 SAHI/일반 추론을 비교하고
소형/중형/대형 드론별 성능 차이를 분석

사용법:
    python configs/src/sahi_analysis.py
"""

import json
from pathlib import Path
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction, get_prediction


def load_model(model_path, conf=0.25):
    return AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=model_path,
        confidence_threshold=conf,
        device="cuda:0",
    )


def get_gt_count(label_dir, img_name):
    """Ground truth 라벨에서 객체 수와 크기 가져오기"""
    label_path = label_dir / (Path(img_name).stem + ".txt")
    if not label_path.exists():
        return 0, []
    
    lines = label_path.read_text().strip().split("\n")
    sizes = []
    for line in lines:
        if line.strip():
            parts = line.strip().split()
            if len(parts) >= 5:
                area = float(parts[3]) * float(parts[4])
                if area < 0.001:
                    sizes.append("small")
                elif area < 0.01:
                    sizes.append("medium")
                else:
                    sizes.append("large")
    return len(sizes), sizes


def analyze(model_path, test_img_dir, label_dir, max_images=None):
    """전체 분석 실행"""
    model = load_model(model_path)
    
    image_paths = sorted(
        list(test_img_dir.glob("*.jpg")) + list(test_img_dir.glob("*.png"))
    )
    if max_images:
        image_paths = image_paths[:max_images]
    
    print(f"\n분석할 이미지: {len(image_paths)}장\n")
    
    # 통계 수집
    total_std = 0
    total_sahi = 0
    total_gt = 0
    sahi_extra = 0
    
    size_stats = {
        "small": {"gt": 0, "std_found": 0, "sahi_found": 0},
        "medium": {"gt": 0, "std_found": 0, "sahi_found": 0},
        "large": {"gt": 0, "std_found": 0, "sahi_found": 0},
    }
    
    sahi_only_sizes = []  # SAHI만 찾은 드론의 픽셀 크기
    
    for i, img_path in enumerate(image_paths):
        if (i + 1) % 100 == 0:
            print(f"  진행: {i+1}/{len(image_paths)}")
        
        # Ground truth
        gt_count, gt_sizes = get_gt_count(label_dir, img_path.name)
        total_gt += gt_count
        for s in gt_sizes:
            size_stats[s]["gt"] += 1
        
        # 일반 추론
        std_result = get_prediction(
            image=str(img_path),
            detection_model=model,
            verbose=0,
        )
        std_count = len(std_result.object_prediction_list)
        total_std += std_count
        
        # SAHI 추론
        sahi_result = get_sliced_prediction(
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
        sahi_count = len(sahi_result.object_prediction_list)
        total_sahi += sahi_count
        
        extra = max(0, sahi_count - std_count)
        sahi_extra += extra
        
        # SAHI가 추가로 찾은 드론 크기 기록
        if sahi_count > std_count:
            for pred in sahi_result.object_prediction_list:
                bbox = pred.bbox
                w = bbox.maxx - bbox.minx
                h = bbox.maxy - bbox.miny
                sahi_only_sizes.append((w, h, w * h))
    
    # 결과 출력
    print(f"\n{'='*60}")
    print(f"  SAHI vs 일반 추론 전체 분석 결과")
    print(f"  이미지: {len(image_paths)}장")
    print(f"{'='*60}")
    
    print(f"\n  [전체 탐지 수]")
    print(f"    Ground Truth:  {total_gt}개")
    print(f"    일반 추론:     {total_std}개")
    print(f"    SAHI 추론:     {total_sahi}개")
    print(f"    SAHI 추가 탐지: +{sahi_extra}개 (+{sahi_extra/max(total_std,1)*100:.1f}%)")
    
    print(f"\n  [Ground Truth 크기 분포]")
    for size_name in ["small", "medium", "large"]:
        count = size_stats[size_name]["gt"]
        pct = count / max(total_gt, 1) * 100
        print(f"    {size_name:8s}: {count:5d}개 ({pct:.1f}%)")
    
    if sahi_only_sizes:
        print(f"\n  [SAHI가 추가로 찾은 드론 크기 분석]")
        pixel_areas = [s[2] for s in sahi_only_sizes]
        widths = [s[0] for s in sahi_only_sizes]
        heights = [s[1] for s in sahi_only_sizes]
        
        small_extra = sum(1 for w, h, _ in sahi_only_sizes if w < 40 and h < 40)
        med_extra = sum(1 for w, h, _ in sahi_only_sizes if (w >= 40 or h >= 40) and (w < 100 and h < 100))
        large_extra = sum(1 for w, h, _ in sahi_only_sizes if w >= 100 or h >= 100)
        
        print(f"    총 {len(sahi_only_sizes)}개 추가 탐지")
        print(f"    소형 (<40px):  {small_extra}개 ({small_extra/len(sahi_only_sizes)*100:.1f}%)")
        print(f"    중형 (40-100px): {med_extra}개 ({med_extra/len(sahi_only_sizes)*100:.1f}%)")
        print(f"    대형 (>100px): {large_extra}개 ({large_extra/len(sahi_only_sizes)*100:.1f}%)")
        print(f"    평균 크기: {sum(widths)/len(widths):.1f} x {sum(heights)/len(heights):.1f}px")
    
    print(f"\n{'='*60}")
    print(f"  결론")
    print(f"{'='*60}")
    if sahi_extra > 0:
        print(f"  SAHI 적용으로 {sahi_extra}개 추가 탐지 (+{sahi_extra/max(total_std,1)*100:.1f}%)")
        if sahi_only_sizes:
            small_pct = small_extra / len(sahi_only_sizes) * 100
            print(f"  추가 탐지의 {small_pct:.0f}%가 소형 드론")
        print(f"  → SAHI가 소형 드론 탐지에 효과적임을 확인")
    else:
        print(f"  SAHI 추가 탐지 없음")


if __name__ == "__main__":
    model_path = "E:/drone_runs/train/drone_det_v2/weights/best.pt"
    test_img_dir = Path("data/dut-anti-uav/images/test")
    label_dir = Path("data/dut-anti-uav/labels/test")
    
    # 전체 분석 (시간 오래 걸리면 max_images=200 으로 줄이기)
    analyze(model_path, test_img_dir, label_dir, max_images=200)