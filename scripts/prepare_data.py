"""
DUT Anti-UAV 데이터셋 전처리 스크립트
======================================
XML (Pascal VOC) → YOLO txt 형식 변환
이미 train/val/test 분할되어 있으므로 그대로 사용

구조:
  data/raw/train/  → img/, xml/
  data/raw/val/    → img/, xml/
  data/raw/test/   → img/, xml/

사용법:
    python scripts/prepare_data.py
"""

import xml.etree.ElementTree as ET
import shutil
from pathlib import Path


def xml_to_yolo(xml_path, class_name="UAV"):
    """
    Pascal VOC XML → YOLO txt 변환

    XML: <xmin>, <ymin>, <xmax>, <ymax> (픽셀 좌표)
    YOLO: class x_center y_center width height (0~1 정규화)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 이미지 크기
    size = root.find("size")
    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)

    yolo_lines = []

    for obj in root.findall("object"):
        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        # YOLO 형식으로 변환
        x_center = ((xmin + xmax) / 2) / img_w
        y_center = ((ymin + ymax) / 2) / img_h
        width = (xmax - xmin) / img_w
        height = (ymax - ymin) / img_h

        # 경계 체크
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        width = max(0, min(1, width))
        height = max(0, min(1, height))

        # 클래스 0 = drone
        yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    return yolo_lines, img_w, img_h


def process_split(split_name, raw_dir, output_dir):
    """한 분할(train/val/test) 처리"""
    img_src = raw_dir / split_name / "img"
    xml_src = raw_dir / split_name / "xml"

    img_dst = output_dir / "images" / split_name
    lbl_dst = output_dir / "labels" / split_name

    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    # XML 파일 목록
    xml_files = sorted(xml_src.glob("*.xml"))
    print(f"\n  [{split_name}] XML 파일: {len(xml_files)}개")

    total_objects = 0
    sizes = []
    skipped = 0

    for xml_path in xml_files:
        # 대응하는 이미지 찾기
        img_name = xml_path.stem + ".jpg"
        img_path = img_src / img_name

        if not img_path.exists():
            # png도 확인
            img_name = xml_path.stem + ".png"
            img_path = img_src / img_name

        if not img_path.exists():
            skipped += 1
            continue

        # XML → YOLO 변환
        yolo_lines, img_w, img_h = xml_to_yolo(xml_path)

        # 이미지 복사
        shutil.copy2(img_path, img_dst / img_path.name)

        # YOLO 라벨 저장
        label_path = lbl_dst / (xml_path.stem + ".txt")
        label_path.write_text("\n".join(yolo_lines))

        total_objects += len(yolo_lines)
        for line in yolo_lines:
            parts = line.split()
            area = float(parts[3]) * float(parts[4])
            sizes.append(area)

    # 통계
    small = sum(1 for s in sizes if s < 0.001)
    medium = sum(1 for s in sizes if 0.001 <= s < 0.01)
    large = sum(1 for s in sizes if s >= 0.01)

    print(f"    이미지 변환: {len(xml_files) - skipped}장")
    print(f"    총 드론 객체: {total_objects}개")
    print(f"    크기 분포 — 소형: {small} | 중형: {medium} | 대형: {large}")
    if skipped:
        print(f"    스킵 (이미지 없음): {skipped}")


def main():
    raw_dir = Path("data/raw")
    output_dir = Path("data/dut-anti-uav")

    print("=" * 60)
    print("  DUT Anti-UAV: XML → YOLO 변환")
    print("=" * 60)

    for split in ["train", "val", "test"]:
        split_path = raw_dir / split
        if not split_path.exists():
            print(f"\n  [경고] {split_path} 폴더 없음, 건너뜀")
            continue
        process_split(split, raw_dir, output_dir)

    print(f"\n{'=' * 60}")
    print(f"  변환 완료! 출력: {output_dir}")
    print(f"{'=' * 60}")
    print(f"\n다음 단계:")
    print(f"  python src/train.py --epochs 50 --batch 8")


if __name__ == "__main__":
    main()