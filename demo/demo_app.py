"""
드론 탐지 데모 앱
==================
이미지를 업로드하면 드론을 탐지하고 결과를 시각화

실행:
    streamlit run configs/src/demo_app.py
"""

import streamlit as st
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import tempfile


# ── 페이지 설정 ──
st.set_page_config(
    page_title="Anti-UAV Detection System",
    page_icon="🎯",
    layout="wide",
)

# ── 스타일 ──
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1B4F72;
        margin-bottom: 0;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #666;
        margin-top: 0;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        border: 1px solid #e9ecef;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1B4F72;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #888;
    }
    .detection-info {
        background: #e8f4f8;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
        border-left: 4px solid #1B4F72;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(model_path):
    """모델 로드 (캐시하여 재로드 방지)"""
    return YOLO(model_path)


def draw_detections(image, results, conf_threshold):
    """탐지 결과를 이미지에 그리기"""
    img = image.copy()
    detections = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            conf = float(box.conf[0])
            if conf < conf_threshold:
                continue

            # 좌표
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w = x2 - x1
            h = y2 - y1

            # 크기 분류
            img_area = img.shape[0] * img.shape[1]
            box_area = w * h
            ratio = box_area / img_area
            if ratio < 0.001:
                size_tag = "Small"
                color = (0, 165, 255)   # 주황
            elif ratio < 0.01:
                size_tag = "Medium"
                color = (0, 200, 0)     # 초록
            else:
                size_tag = "Large"
                color = (255, 50, 50)   # 파랑 (BGR)

            # 박스 그리기
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # 라벨 배경
            label = f"Drone {conf:.0%}  [{size_tag}]"
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(img, (x1, y1 - label_h - 10), (x1 + label_w + 8, y1), color, -1)
            cv2.putText(img, label, (x1 + 4, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            detections.append({
                "conf": conf,
                "bbox": (x1, y1, x2, y2),
                "size": (w, h),
                "size_tag": size_tag,
                "area_ratio": ratio,
            })

    return img, detections


def main():
    # ── 헤더 ──
    st.markdown('<p class="main-title">Anti-UAV Detection System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">YOLOv8m  |  DUT Anti-UAV Dataset  |  Real-time Drone Detection</p>', unsafe_allow_html=True)
    st.markdown("---")

    # ── 사이드바 ──
    with st.sidebar:
        st.header("Settings")

        model_path = st.text_input(
            "Model path",
            value="E:/drone_runs/train/drone_det_v2/weights/best.pt",
        )

        conf_threshold = st.slider(
            "Confidence threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.46,
            step=0.01,
            help="F1 optimal: 0.461"
        )

        st.markdown("---")
        st.markdown("**Threshold guide**")
        st.markdown("""
        - **0.2~0.3**: High recall (military)
        - **0.46**: F1 optimal (balanced)
        - **0.6~0.8**: High precision (civilian)
        """)

        st.markdown("---")
        st.markdown("**Model performance**")
        st.markdown("""
        - mAP@50: 91.1%
        - mAP@50-95: 54.9%
        - Precision: 95.9%
        - Recall: 86.1%
        - Speed: 5ms/image
        """)

    # ── 모델 로드 ──
    try:
        model = load_model(model_path)
        st.sidebar.success("Model loaded")
    except Exception as e:
        st.error(f"Model load failed: {e}")
        return

    # ── 메인 영역 ──
    tab1, tab2 = st.tabs(["Image Detection", "Batch Detection"])

    # ── Tab 1: 단일 이미지 ──
    with tab1:
        uploaded_file = st.file_uploader(
            "Upload drone image",
            type=["jpg", "jpeg", "png", "bmp"],
            help="Upload an image to detect drones"
        )

        # 샘플 이미지 버튼
        sample_dir = Path("data/dut-anti-uav/images/test")
        if sample_dir.exists():
            sample_images = sorted(sample_dir.glob("*.jpg"))[:10]
            if sample_images:
                st.markdown("**Or try a sample image:**")
                cols = st.columns(min(5, len(sample_images)))
                for i, img_path in enumerate(sample_images[:5]):
                    with cols[i]:
                        if st.button(img_path.name, key=f"sample_{i}"):
                            st.session_state["sample_path"] = str(img_path)

        # 이미지 처리
        image = None
        source_name = ""

        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            source_name = uploaded_file.name
        elif "sample_path" in st.session_state:
            image = cv2.imread(st.session_state["sample_path"])
            source_name = Path(st.session_state["sample_path"]).name

        if image is not None:
            # 추론
            results = model(image, verbose=False)
            result_img, detections = draw_detections(image, results, conf_threshold)

            # 결과 표시
            col1, col2 = st.columns([3, 1])

            with col1:
                st.image(
                    cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB),
                    caption=f"Detection result: {source_name}",
                    use_container_width=True,
                )

            with col2:
                # 메트릭 카드
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{len(detections)}</div>
                    <div class="metric-label">Drones detected</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("")

                # 이미지 정보
                h, w = image.shape[:2]
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{w} x {h}</div>
                    <div class="metric-label">Image size (px)</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("")

                # 크기별 카운트
                small_count = sum(1 for d in detections if d["size_tag"] == "Small")
                med_count = sum(1 for d in detections if d["size_tag"] == "Medium")
                large_count = sum(1 for d in detections if d["size_tag"] == "Large")

                if small_count:
                    st.markdown(f"🟠 Small: **{small_count}**")
                if med_count:
                    st.markdown(f"🟢 Medium: **{med_count}**")
                if large_count:
                    st.markdown(f"🔵 Large: **{large_count}**")

            # 탐지 상세 정보
            if detections:
                st.markdown("### Detection details")
                for i, det in enumerate(detections):
                    x1, y1, x2, y2 = det["bbox"]
                    w, h = det["size"]
                    st.markdown(f"""
                    <div class="detection-info">
                        <strong>#{i+1}</strong> &nbsp;
                        Confidence: <strong>{det['conf']:.1%}</strong> &nbsp; | &nbsp;
                        Size: <strong>{w} x {h}px</strong> ({det['size_tag']}) &nbsp; | &nbsp;
                        Position: ({x1}, {y1}) → ({x2}, {y2})
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No drones detected at the current confidence threshold. Try lowering it in the sidebar.")

    # ── Tab 2: 배치 처리 ──
    with tab2:
        uploaded_files = st.file_uploader(
            "Upload multiple images",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="batch_upload",
        )

        if uploaded_files:
            st.markdown(f"**Processing {len(uploaded_files)} images...**")

            total_drones = 0
            batch_results = []

            progress = st.progress(0)

            for idx, file in enumerate(uploaded_files):
                file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                results = model(img, verbose=False)
                result_img, detections = draw_detections(img, results, conf_threshold)

                total_drones += len(detections)
                batch_results.append((file.name, result_img, detections))
                progress.progress((idx + 1) / len(uploaded_files))

            progress.empty()

            # 요약
            st.markdown(f"""
            <div class="metric-card" style="margin-bottom: 16px;">
                <div class="metric-value">{total_drones}</div>
                <div class="metric-label">Total drones in {len(uploaded_files)} images</div>
            </div>
            """, unsafe_allow_html=True)

            # 결과 그리드
            cols = st.columns(min(3, len(batch_results)))
            for i, (name, img, dets) in enumerate(batch_results):
                with cols[i % 3]:
                    st.image(
                        cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                        caption=f"{name} ({len(dets)} drones)",
                        use_container_width=True,
                    )


if __name__ == "__main__":
    main()