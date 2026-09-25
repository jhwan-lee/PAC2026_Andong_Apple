"""
PAC 2026 안동청과 미션 - Intel RealSense D435 + YOLO11 사과 품질 검출 및 3D 공간좌표 추출기
========================================================================================
작성자: 이재환 & 아티 (A-ti)
기능:
1. D435 카메라 실시간 RGB + Depth 동기화 스트리밍 (1280x720 @ 30fps)
2. 하드웨어 시차 보정 (rs.align) 적용 및 카메라 내부 파라미터(Intrinsics) 자동 획득
3. YOLO11 사과 품질 판정 모델(best.pt) 실시간 추론 (상/중/결점 분류)
4. 사과 중심점 (cx, cy) 주변 5x5 중앙값 필터링 기반 정밀 깊이(Z) 측정
5. rs2_deproject_pixel_to_point를 통한 실제 로봇 3차원 공간 좌표 (X, Y, Z mm) 변환
6. CIE-Lab 물리적 착색도(%) 하이브리드 교차 검증
7. 실시간 GUI 화면 표시 및 's' 키 입력 시 스냅샷 저장, 'q' 또는 ESC 키 입력 시 종료
"""

import os
import sys
import time
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

# 로컬 품질 검사기 import
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

try:
    from apple_quality_detector import AppleQualityDetector
    HAVE_PHYS_DETECTOR = True
except ImportError:
    HAVE_PHYS_DETECTOR = False


class RealSenseYOLOAppleDetector:
    CLASS_NAMES_KOREAN = {
        0: "상 (grade_high)",
        1: "중 (grade_mid)",
        2: "결점 (defect)"
    }
    CLASS_COLORS = {
        0: (0, 220, 0),     # 녹색 (상급)
        1: (0, 215, 255),   # 노란색 (중급)
        2: (0, 0, 255)      # 빨간색 (결점)
    }

    def __init__(
        self,
        weights_path: str = "/home/jaehwan/Documents/PAC2026_Andong_Apple/runs/detect/runs/train_apple/apple_quality_v1-2/weights/best.pt",
        backup_coco_weights: str = "/home/jaehwan/Documents/PAC2026_Andong_Apple/yolo_quality/yolo11n.pt",
        conf_thresh: float = 0.20,
        width: int = 1280,
        height: int = 720,
        fps: int = 30
    ):
        self.conf_thresh = conf_thresh
        self.width = width
        self.height = height
        self.fps = fps

        # 1. 모델 로드
        print(f"📦 [1/3] YOLO 가중치 로드 중: {weights_path}")
        if os.path.exists(weights_path):
            self.model = YOLO(weights_path)
            self.is_custom_quality = True
            print("   ✅ 커스텀 사과 품질 판정 모델(v1-2 best.pt) 로드 성공")
        else:
            print(f"   ⚠️ 커스텀 가중치 없음 -> 기본 COCO 모델({backup_coco_weights}) 로드")
            self.model = YOLO(backup_coco_weights)
            self.is_custom_quality = False

        # 백업 COCO 모델도 함께 준비 (사과가 일반 사과 클래스로만 잡힐 경우 대비)
        self.coco_model = None
        if os.path.exists(backup_coco_weights):
            self.coco_model = YOLO(backup_coco_weights)

        # 물리 비전 검사기
        self.phys_detector = AppleQualityDetector() if HAVE_PHYS_DETECTOR else None

        # 2. RealSense 파이프라인 설정
        print("📷 [2/3] Intel RealSense D435 파이프라인 초기화 중...")
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)

        # Depth to Color 하드웨어 시차 정렬기
        self.align = rs.align(rs.stream.color)
        self.colorizer = rs.colorizer()

        # 파이프라인 시작
        self.profile = self.pipeline.start(self.config)
        self.depth_intrinsics = None
        print("   ✅ RealSense D435 스트리밍 시작 완료 (1280x720 @ 30fps)")

    def get_robust_depth_mm(self, depth_frame: rs.depth_frame, cx: int, cy: int, patch_radius: int = 4) -> float:
        """
        중심점 주변 patch_radius 반경의 뎁스 값을 샘플링하여 0(노이즈)을 제외한 중앙값(Median) 반환
        """
        w = depth_frame.get_width()
        h = depth_frame.get_height()
        
        depth_samples = []
        for dy in range(-patch_radius, patch_radius + 1):
            ny = cy + dy
            if ny < 0 or ny >= h:
                continue
            for dx in range(-patch_radius, patch_radius + 1):
                nx = cx + dx
                if nx < 0 or nx >= w:
                    continue
                d = depth_frame.get_distance(nx, ny)  # 미터 단위
                if d > 0.15:  # D435 최소 데드존 초과 유효값만
                    depth_samples.append(d)

        if len(depth_samples) > 0:
            return float(np.median(depth_samples)) * 1000.0  # mm 변환
        
        # 샘플이 없을 경우 직접 픽셀 값 확인
        direct = depth_frame.get_distance(cx, cy)
        return float(direct) * 1000.0 if direct > 0 else 0.0

    def deproject_to_3d(self, cx: int, cy: int, depth_mm: float) -> Tuple[float, float, float]:
        """
        2D 픽셀 좌표 (cx, cy)와 깊이(mm) -> 카메라 3차원 공간 좌표 (X, Y, Z mm) 변환
        """
        if self.depth_intrinsics is None or depth_mm <= 0:
            return (0.0, 0.0, 0.0)
        
        depth_m = depth_mm / 1000.0
        pt_3d = rs.rs2_deproject_pixel_to_point(self.depth_intrinsics, [cx, cy], depth_m)
        return (round(pt_3d[0] * 1000.0, 1), round(pt_3d[1] * 1000.0, 1), round(depth_mm, 1))

    def run_detection(self, max_frames: Optional[int] = None, save_path: str = "d435_yolo_detection_result.png"):
        """
        실시간 검출 루프 실행
        """
        print("🚀 [3/3] 사과 검출 및 3D 계측 루프 가동 시작!")
        print("   * 's' 키: 현재 프레임 캡처 저장")
        print("   * 'q' 또는 ESC 키: 종료")

        frame_count = 0
        last_save_time = 0

        try:
            while True:
                frames = self.pipeline.wait_for_frames(timeout_ms=5000)
                aligned_frames = self.align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()

                if not depth_frame or not color_frame:
                    continue

                if self.depth_intrinsics is None:
                    self.depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

                color_img = np.asanyarray(color_frame.get_data())
                depth_colormap = np.asanyarray(self.colorizer.colorize(depth_frame).get_data())

                # 1. 커스텀 품질 모델 추론
                yolo_res = self.model.predict(color_img, conf=self.conf_thresh, verbose=False)[0]
                boxes = yolo_res.boxes

                # 만약 커스텀 모델에서 사과가 하나도 안 잡혔는데 COCO 모델이 있으면 COCO 사과(class 47) 교차 검증
                detected_items = []
                if len(boxes) > 0:
                    for box in boxes:
                        cls_id = int(box.cls[0].item())
                        conf = float(box.conf[0].item())
                        cls_name = yolo_res.names.get(cls_id, str(cls_id))
                        grade_kor = self.CLASS_NAMES_KOREAN.get(cls_id, cls_name)
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        detected_items.append((cls_id, conf, grade_kor, (x1, y1, x2, y2)))
                elif self.coco_model is not None:
                    coco_res = self.coco_model.predict(color_img, conf=self.conf_thresh, verbose=False)[0]
                    for box in coco_res.boxes:
                        cls_id = int(box.cls[0].item())
                        cls_name = coco_res.names.get(cls_id, str(cls_id))
                        # COCO class 47: apple
                        if cls_name == "apple" or cls_id == 47:
                            conf = float(box.conf[0].item())
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            detected_items.append((0, conf, "사과(일반)", (x1, y1, x2, y2)))

                annotated_img = color_img.copy()
                detection_summary = []

                # 검출된 사과들 3D 좌표 및 품질 가공
                for idx, (cls_id, conf, grade_kor, (x1, y1, x2, y2)) in enumerate(detected_items):
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    radius = int((x2 - x1 + y2 - y1) / 4)

                    # 깊이 측정 및 3D 좌표 변환
                    depth_mm = self.get_robust_depth_mm(depth_frame, cx, cy)
                    X_mm, Y_mm, Z_mm = self.deproject_to_3d(cx, cy, depth_mm)

                    # 물리적 착색도 분석
                    phys_info = ""
                    crop = color_img[max(0, y1):min(self.height, y2), max(0, x1):min(self.width, x2)]
                    if self.phys_detector and crop.size > 0:
                        q_res = self.phys_detector.inspect(crop)
                        phys_info = f"Red:{q_res.color_ratio:.0f}%"

                    box_color = self.CLASS_COLORS.get(cls_id, (0, 255, 255))

                    # 1) 바운딩 박스 & 중심 십자선
                    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), box_color, 3)
                    cv2.drawMarker(annotated_img, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
                    cv2.circle(annotated_img, (cx, cy), radius, box_color, 1)

                    # 2) 텍스트 오버레이
                    label_top = f"#{idx+1} [{grade_kor}] {conf*100:.1f}% {phys_info}"
                    if Z_mm > 0:
                        label_3d = f"3D: X={X_mm:+.0f}, Y={Y_mm:+.0f}, Z={Z_mm:.0f} mm"
                    else:
                        label_3d = "3D: Z < 20cm (Deadzone)"

                    # 라벨 배경 박스
                    cv2.rectangle(annotated_img, (x1, max(0, y1 - 48)), (x1 + max(len(label_top), len(label_3d)) * 11, y1), (20, 20, 20), -1)
                    cv2.putText(annotated_img, label_top, (x1 + 4, y1 - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(annotated_img, label_3d, (x1 + 4, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

                    detection_summary.append({
                        "id": idx + 1,
                        "grade": grade_kor,
                        "conf": round(conf, 3),
                        "box": [x1, y1, x2, y2],
                        "center_px": [cx, cy],
                        "coord_3d_mm": [X_mm, Y_mm, Z_mm],
                        "phys_color_ratio": phys_info
                    })

                # 상단 헤더 HUD 정보
                hud_text = f"PAC2026 Apple Detector | Apples Detected: {len(detected_items)} | Res: {self.width}x{self.height}"
                cv2.rectangle(annotated_img, (0, 0), (self.width, 36), (15, 15, 15), -1)
                cv2.putText(annotated_img, hud_text, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)

                # 우측 하단에 Depth Colormap 미니 맵 오버레이 (320x180)
                mini_depth = cv2.resize(depth_colormap, (320, 180))
                annotated_img[self.height - 185:self.height - 5, self.width - 325:self.width - 5] = mini_depth
                cv2.rectangle(annotated_img, (self.width - 325, self.height - 185), (self.width - 5, self.height - 5), (0, 255, 200), 1)
                cv2.putText(annotated_img, "Depth Map", (self.width - 315, self.height - 165), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # 자동 저장 (처음 사과가 검출되거나 1초 주기)
                current_time = time.time()
                if (len(detected_items) > 0 and current_time - last_save_time > 2.0) or frame_count == 10:
                    cv2.imwrite(save_path, annotated_img)
                    last_save_time = current_time
                    if len(detected_items) > 0:
                        print(f"🍏 [프레임 {frame_count}] 사과 {len(detected_items)}개 감지됨! 스냅샷 저장 -> {save_path}")
                        for item in detection_summary:
                            print(f"   * 타겟 #{item['id']}: 등급={item['grade']}, 신뢰도={item['conf']*100:.1f}%, 3D좌표={item['coord_3d_mm']}")

                # GUI 창 표시 (X11 환경)
                cv2.imshow("PAC 2026 Apple Quality & 3D Locator (Intel D435)", annotated_img)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
                elif key == ord('s'):
                    snapshot_name = f"snapshot_apple_{int(time.time())}.png"
                    cv2.imwrite(snapshot_name, annotated_img)
                    print(f"📸 수동 스냅샷 저장 완료: {snapshot_name}")

                frame_count += 1
                if max_frames and frame_count >= max_frames:
                    cv2.imwrite(save_path, annotated_img)
                    break

        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()
            print("🛑 스트리밍 파이프라인 정상 종료 완료.")


def main():
    detector = RealSenseYOLOAppleDetector()
    max_f = None
    if len(sys.argv) > 1 and sys.argv[1] == "--verify-frames":
        max_f = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    save_path = sys.argv[3] if len(sys.argv) > 3 else "d435_yolo_detection_result.png"
    detector.run_detection(max_frames=max_f, save_path=save_path)


if __name__ == "__main__":
    main()
