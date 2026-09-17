"""
PAC 2026 분과 1 안동청과 미션 - 사과 위치 탐색 및 로봇 좌표 변환 모듈
=====================================================================
기능:
1. 카메라 화각 내 트레이 영역 및 사과 5개 위치(2D 중심점, 반경) 동시 검출
2. 겹치거나 인접한 사과의 개별 인스턴스 분리
3. 카메라 픽셀 좌표 (u, v) -> 로봇 베이스 좌표계 (X, Y, Z mm) 변환 (Hand-Eye Calibration 매핑)
4. 파지 사이클 타임 단축을 위한 최적 픽업 순서(TSP/정렬) 계산
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class AppleTarget:
    target_id: int
    center_px: Tuple[int, int]     # (u, v) 픽셀 좌표
    radius_px: int                 # 사과 반경 픽셀
    robot_xyz: Tuple[float, float, float]  # 로봇 베이스 좌표계 (X, Y, Z mm)
    crop_img: np.ndarray           # 해당 사과 영역 크롭 이미지
    mask: np.ndarray               # 개별 사과 마스크


class AppleLocator:
    """
    트레이 위 사과 5개 탐색 및 3D 로봇 파지 좌표 변환기
    """

    def __init__(
        self,
        pixel_to_mm_ratio: float = 0.5,      # 1픽셀 당 실제 거리(mm) (기본값, 캘리브레이션으로 갱신)
        camera_center_robot: Tuple[float, float, float] = (300.0, 0.0, 450.0), # 카메라 광학 중심의 로봇 좌표
        pickup_z_height: float = 35.0        # 사과 파지 높이 (트레이 바닥면 기준 Z mm)
    ):
        self.pixel_to_mm = pixel_to_mm_ratio
        self.cam_robot_origin = camera_center_robot
        self.pickup_z = pickup_z_height
        self.homography_matrix: Optional[np.ndarray] = None

    def set_calibration_matrix(self, H: np.ndarray):
        """
        4점 매핑 또는 Hand-Eye Calibration으로 얻은 3x3 호모그래피 행렬 설정
        """
        self.homography_matrix = H

    def detect_apples(
        self,
        bgr_img: np.ndarray,
        max_targets: int = 5
    ) -> List[AppleTarget]:
        """
        트레이 내 사과 탐색 및 개별 타겟 인스턴스 생성
        """
        hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        h, w = bgr_img.shape[:2]

        # 붉은색 및 황색/녹색 과실 영역 검출 마스크
        red1 = cv2.inRange(hsv, np.array([0, 50, 40]), np.array([15, 255, 255]))
        red2 = cv2.inRange(hsv, np.array([160, 50, 40]), np.array([180, 255, 255]))
        yellow_green = cv2.inRange(hsv, np.array([20, 40, 40]), np.array([45, 255, 255]))
        apple_color_mask = red1 | red2 | yellow_green

        # 노이즈 제거
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        clean_mask = cv2.morphologyEx(apple_color_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # Watershed 또는 허프 서클을 보조로 결합한 중심점 탐색
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=45,
            param1=50,
            param2=25,
            minRadius=25,
            maxRadius=120
        )

        detected_apples: List[AppleTarget] = []

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i, c in enumerate(circles[0, :]):
                cx, cy, r = int(c[0]), int(c[1]), int(c[2])

                # 마스크 내 유효 픽셀 비율 확인 (배경 오검출 방지)
                y1, y2 = max(0, cy - r), min(h, cy + r)
                x1, x2 = max(0, cx - r), min(w, cx + r)
                roi_mask = clean_mask[y1:y2, x1:x2]

                if roi_mask.size > 0 and np.count_nonzero(roi_mask) / roi_mask.size > 0.25:
                    # 크롭 이미지
                    crop = bgr_img[y1:y2, x1:x2].copy()

                    # 로봇 좌표 계산
                    robot_x, robot_y, robot_z = self.pixel_to_robot_xyz(cx, cy, bgr_img.shape)

                    detected_apples.append(
                        AppleTarget(
                            target_id=len(detected_apples) + 1,
                            center_px=(cx, cy),
                            radius_px=r,
                            robot_xyz=(robot_x, robot_y, robot_z),
                            crop_img=crop,
                            mask=roi_mask
                        )
                    )
                if len(detected_apples) >= max_targets:
                    break

        # 허프 서클 실패 시 컨투어 기반 백업
        if len(detected_apples) == 0:
            contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            sorted_c = sorted(contours, key=cv2.contourArea, reverse=True)[:max_targets]

            for i, c in enumerate(sorted_c):
                if cv2.contourArea(c) < 800:
                    continue
                (cx, cy), radius = cv2.minEnclosingCircle(c)
                cx, cy, r = int(cx), int(cy), int(radius)
                y1, y2 = max(0, cy - r), min(h, cy + r)
                x1, x2 = max(0, cx - r), min(w, cx + r)
                crop = bgr_img[y1:y2, x1:x2].copy()
                robot_x, robot_y, robot_z = self.pixel_to_robot_xyz(cx, cy, bgr_img.shape)

                detected_apples.append(
                    AppleTarget(
                        target_id=i + 1,
                        center_px=(cx, cy),
                        radius_px=r,
                        robot_xyz=(robot_x, robot_y, robot_z),
                        crop_img=crop,
                        mask=clean_mask[y1:y2, x1:x2]
                    )
                )

        return detected_apples

    def pixel_to_robot_xyz(
        self,
        u: int,
        v: int,
        img_shape: Tuple[int, ...]
    ) -> Tuple[float, float, float]:
        """
        카메라 픽셀 (u, v) -> 로봇 좌표 (X, Y, Z mm) 변환
        """
        if self.homography_matrix is not None:
            # 3x3 변환 행렬 적용
            pt = np.array([[[u, v]]], dtype=np.float32)
            dst = cv2.perspectiveTransform(pt, self.homography_matrix)
            rx, ry = float(dst[0][0][0]), float(dst[0][0][1])
            return round(rx, 1), round(ry, 1), self.pickup_z

        # 기본 기하 매핑 (카메라 중심 기준 상대 거리)
        img_h, img_w = img_shape[:2]
        center_u, center_v = img_w / 2.0, img_h / 2.0

        # 카메라 축과 로봇 좌표축 정렬 매핑 (X: 전방, Y: 좌측, Z: 상방)
        dx_mm = (v - center_v) * self.pixel_to_mm  # 세로축(v) 증가 = 로봇 앞쪽(X)
        dy_mm = -(u - center_u) * self.pixel_to_mm # 가로축(u) 증가 = 로봇 우측(-Y)

        rx = self.cam_robot_origin[0] + dx_mm
        ry = self.cam_robot_origin[1] + dy_mm
        rz = self.pickup_z

        return round(rx, 1), round(ry, 1), round(rz, 1)

    def sort_targets_for_cycle_time(
        self,
        targets: List[AppleTarget],
        robot_home_xy: Tuple[float, float] = (200.0, 0.0)
    ) -> List[AppleTarget]:
        """
        로봇 암 동선을 최소화하기 위한 최근접 순서(Greedy Path) 정렬
        """
        if not targets:
            return []

        remaining = targets.copy()
        sorted_targets = []
        curr_pos = robot_home_xy

        while remaining:
            # 현재 위치와 가장 가까운 타겟 선택
            nearest = min(
                remaining,
                key=lambda t: (t.robot_xyz[0] - curr_pos[0])**2 + (t.robot_xyz[1] - curr_pos[1])**2
            )
            sorted_targets.append(nearest)
            curr_pos = (nearest.robot_xyz[0], nearest.robot_xyz[1])
            remaining.remove(nearest)

        return sorted_targets
