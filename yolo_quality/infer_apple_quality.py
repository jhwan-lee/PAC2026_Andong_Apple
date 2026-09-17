"""
PAC 2026 안동청과 미션 - 사과 품질 실시간 추론 및 팀 연동 인터페이스
================================================================================
작성자: 이재환 (YOLO 사과 학습 & 품질 구분 파트장)
역할:
1. 학습된 YOLO 모델(best.pt)을 로드하여 실시간 사과 검출 및 등급 판정
2. 2번 파트(비전 좌표팀)를 위해 각 사과의 바운딩 박스 및 중심점 (cx, cy) 제공
3. 3번 파트(로봇 제어팀)를 위해 등급별(상/중/결점) 목적지 상자 타겟 큐(Queue) 제공
4. [확장 가산점] YOLO 판정 결과 + CIE-Lab 착색도 정량 수치 융합 하이브리드 모드 지원
"""

import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import cv2
import numpy as np

# 필요 시 상위 디렉터리 모듈 import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from apple_quality_detector import AppleQualityDetector
    HAVE_DETECTOR = True
except ImportError:
    HAVE_DETECTOR = False


@dataclass
class AppleDetectionItem:
    target_id: int
    class_id: int               # 0: grade_high, 1: grade_mid, 2: defect
    class_name: str             # "grade_high", "grade_mid", "defect"
    grade_korean: str           # "상", "중", "결점(리젝트)"
    confidence: float           # YOLO 신뢰도 (0.0 ~ 1.0)
    box_xyxy: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    center_px: Tuple[int, int]  # (cx, cy) -> 비전 좌표팀이 사용할 중심 좌표
    radius_px: int              # 반경 (너비/높이 평균 / 2)
    color_ratio_phys: float     # CIE-Lab 물리적 착색률 (하이브리드 모드 시)
    crop_img: np.ndarray        # 사과 영역 크롭 이미지


class AppleYOLOClassifier:
    """
    사과 품질 판별 YOLO 추론 엔진
    - 비전 좌표팀 및 로봇 제어팀과의 표준 연동 인터페이스를 제공합니다.
    """

    CLASS_MAP_KOREAN = {
        0: "상",
        1: "중",
        2: "결점(리젝트)"
    }

    def __init__(
        self,
        weights_path: Optional[str] = None,
        conf_threshold: float = 0.5,
        use_hybrid_color: bool = True
    ):
        self.conf_thresh = conf_threshold
        self.use_hybrid = use_hybrid_color and HAVE_DETECTOR
        self.model = None

        # 가중치 자동 탐색 (지정되지 않았거나 기본값일 때 최신 best.pt 검색)
        if weights_path is None or not os.path.exists(weights_path):
            candidates = [
                "/home/jaehwan/Documents/PAC2026_Andong_Apple/runs/detect/runs/train_apple/apple_quality_v1-2/weights/best.pt",
                "/home/jaehwan/Documents/PAC2026_Andong_Apple/runs/detect/runs/train_apple/apple_quality_v1/weights/best.pt",
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs/train_apple/apple_quality_v1/weights/best.pt")
            ]
            for c in candidates:
                if os.path.exists(c):
                    weights_path = c
                    break

        if weights_path and os.path.exists(weights_path):
            from ultralytics import YOLO
            self.model = YOLO(weights_path)
            print(f"[AppleYOLO] 최적 모델 가중치 로드 완료: {weights_path}")
        else:
            print("[AppleYOLO] 가중치 파일 미지정 또는 없음. 룰베이스 백업 모드로 동작합니다.")

        if self.use_hybrid:
            self.phys_detector = AppleQualityDetector()

    def predict_tray(self, bgr_img: np.ndarray) -> List[AppleDetectionItem]:
        """
        트레이 전체 영상을 입력받아 사과 5개의 위치 및 등급을 동시 판별
        """
        results: List[AppleDetectionItem] = []
        h, w = bgr_img.shape[:2]

        if self.model is not None:
            # YOLO 실시간 추론
            yolo_out = self.model.predict(bgr_img, conf=self.conf_thresh, verbose=False)[0]
            boxes = yolo_out.boxes

            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                cls_name = yolo_out.names.get(cls_id, str(cls_id))
                korean_grade = self.CLASS_MAP_KOREAN.get(cls_id, "중")

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                r = int((x2 - x1 + y2 - y1) / 4)

                # 바운딩 박스 클리핑
                x1_c, y1_c = max(0, x1), max(0, y1)
                x2_c, y2_c = min(w, x2), min(h, y2)
                crop = bgr_img[y1_c:y2_c, x1_c:x2_c].copy()

                # 물리적 착색률 하이브리드 교차 검증
                phys_ratio = 0.0
                if self.use_hybrid and crop.size > 0:
                    q_res = self.phys_detector.inspect(crop)
                    phys_ratio = q_res.color_ratio

                results.append(
                    AppleDetectionItem(
                        target_id=i + 1,
                        class_id=cls_id,
                        class_name=cls_name,
                        grade_korean=korean_grade,
                        confidence=round(conf, 3),
                        box_xyxy=(x1, y1, x2, y2),
                        center_px=(cx, cy),
                        radius_px=r,
                        color_ratio_phys=phys_ratio,
                        crop_img=crop
                    )
                )
        else:
            # 모델 가중치 파일이 아직 없을 때 개발 편의를 위한 룰베이스 폴백
            print("[AppleYOLO] 사전 가중치 대기 중 -> 물리 비전 엔진으로 대체 판정 수행")
            from apple_locator import AppleLocator
            locator = AppleLocator()
            targets = locator.detect_apples(bgr_img, max_targets=5)
            for i, t in enumerate(targets):
                cx, cy = t.center_px
                r = t.radius_px
                x1, y1 = max(0, cx - r), max(0, cy - r)
                x2, y2 = min(w, cx + r), min(h, cy + r)
                
                phys_ratio = 0.0
                grade_kor = "중"
                cls_id = 1
                if self.use_hybrid and t.crop_img.size > 0:
                    q_res = self.phys_detector.inspect(t.crop_img)
                    phys_ratio = q_res.color_ratio
                    grade_kor = q_res.grade_2_tier
                    cls_id = 0 if grade_kor == "상" else 1

                results.append(
                    AppleDetectionItem(
                        target_id=i + 1,
                        class_id=cls_id,
                        class_name="grade_high" if cls_id == 0 else "grade_mid",
                        grade_korean=grade_kor,
                        confidence=0.95,
                        box_xyxy=(x1, y1, x2, y2),
                        center_px=(cx, cy),
                        radius_px=r,
                        color_ratio_phys=phys_ratio,
                        crop_img=t.crop_img
                    )
                )

        return results

    def draw_detections(self, bgr_img: np.ndarray, items: List[AppleDetectionItem]) -> np.ndarray:
        """
        시각화 오버레이 (UI 및 디버깅용)
        """
        vis = bgr_img.copy()
        for item in items:
            x1, y1, x2, y2 = item.box_xyxy
            cx, cy = item.center_px
            color = (0, 220, 0) if item.class_id == 0 else (0, 165, 255)
            if item.class_id == 2:
                color = (0, 0, 255)

            # 박스 및 중심점
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.circle(vis, (cx, cy), 4, (255, 255, 255), -1)

            # 라벨 텍스트
            label = f"[{item.grade_korean}] {item.confidence:.2f}"
            if item.color_ratio_phys > 0:
                label += f" ({item.color_ratio_phys:.0f}%)"

            cv2.putText(vis, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            cv2.putText(vis, f"Center: ({cx}, {cy})", (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

        return vis
