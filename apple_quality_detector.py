"""
PAC 2026 분과 1 안동청과 미션 - 사과 외관 품질 판별 모듈
============================================================
기능:
1. 조명 변화 및 반사광에 강인한 CIE-Lab 및 HSV 기반 착색 비율(Color Coverage Ratio) 계산
2. 반사광(Specular Highlight) 및 어두운 그림자 영역 노이즈 필터링
3. 기본 미션 2등급(상/중) 및 확장 미션 3등급(특/상/보통/리젝트) 자동 분류
4. 파지 후 로봇 손목 회전에 대응하는 다면 검사(Multi-view) 종합 점수 산출
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class QualityResult:
    grade_2_tier: str       # "상", "중" (기본 미션)
    grade_3_tier: str       # "특", "상", "보통", "불합격(리젝트)" (확장 미션)
    color_ratio: float      # 착색 비율 (0.0 ~ 100.0%)
    defect_ratio: float     # 흠집/결점 비율 (0.0 ~ 100.0%)
    mean_a_star: float      # CIE-Lab a* 채널 평균 (적색도 강도)
    is_valid: bool          # 사과 검출 유효성
    details: Dict[str, float]


class AppleQualityDetector:
    """
    사과 품질 자동 판별 엔진
    - 국립농산물품질관리원 농산물 표준규격(사과) 착색비율 기준 준용
    - 어두운 공장 환경 및 반사광에 강인하도록 L* 채널 분리 및 적색도(a* + HSV) 복합 판정
    """

    def __init__(
        self,
        min_red_ratio_premium: float = 70.0,   # 특(상-밝은색): 70% 이상
        min_red_ratio_standard: float = 50.0,  # 상: 50% 이상
        min_red_ratio_commercial: float = 30.0 # 보통(중-어두운색): 30% 이상
    ):
        self.min_red_premium = min_red_ratio_premium
        self.min_red_standard = min_red_ratio_standard
        self.min_red_commercial = min_red_ratio_commercial

    def create_apple_mask(self, bgr_img: np.ndarray) -> np.ndarray:
        """
        배경(트레이, 컨베이어)을 분리하고 사과 영역만 마스킹
        크롭 이미지인 경우 중심 원형 마스크를 기본 적용하고, 조명 변화에 강인하게 색상 필터링
        """
        h, w = bgr_img.shape[:2]
        hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        
        # 크롭 이미지 특성 활용: 중심 기준 원형 마스크
        circle_mask = np.zeros((h, w), dtype=np.uint8)
        cx, cy = w // 2, h // 2
        r = int(min(w, h) * 0.48)
        cv2.circle(circle_mask, (cx, cy), r, 255, -1)

        # 색상 마스크 (적색, 암적색, 황녹색 등 과실 색상 범위)
        red1 = cv2.inRange(hsv, np.array([0, 30, 20]), np.array([20, 255, 255]))
        red2 = cv2.inRange(hsv, np.array([155, 30, 20]), np.array([180, 255, 255]))
        green_yellow = cv2.inRange(hsv, np.array([20, 25, 20]), np.array([60, 255, 255]))
        color_mask = red1 | red2 | green_yellow

        # 원형 마스크와 색상 마스크 결합
        combined = cv2.bitwise_and(circle_mask, color_mask)

        # 모폴로지 연산으로 사과 내부 홀 채우기
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 만약 색상 마스크가 너무 적으면(매우 어둡거나 특이 색상) 원형 마스크를 fallback으로 사용
        if np.count_nonzero(mask) < (np.pi * r * r * 0.3):
            return circle_mask

        return mask

    def filter_specular_highlights(self, bgr_img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        조명 반사로 하얗게 날아간 영역(Specular Highlight)을 마스크에서 제외
        """
        hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        val = hsv[:, :, 2]
        sat = hsv[:, :, 1]

        # 밝기가 매우 높고 채도가 낮은 픽셀 = 반사광
        highlight = (val > 235) & (sat < 50) & (mask > 0)
        valid_mask = mask.copy()
        valid_mask[highlight] = 0
        return valid_mask

    def analyze_color_coverage(
        self,
        bgr_img: np.ndarray,
        apple_mask: Optional[np.ndarray] = None
    ) -> Tuple[float, float, np.ndarray]:
        """
        CIE-Lab 및 HSV 색공간을 결합한 정밀 착색 비율 계산
        반환: (착색비율 %, 평균 a* 적색강도, 적색 영역 시각화 마스크)
        """
        if apple_mask is None:
            apple_mask = self.create_apple_mask(bgr_img)

        valid_mask = self.filter_specular_highlights(bgr_img, apple_mask)
        total_pixels = np.count_nonzero(valid_mask)
        if total_pixels == 0:
            return 0.0, 0.0, np.zeros_like(apple_mask)

        # 1. CIE-Lab 색공간 변환 (조명 명도 L 성분 분리)
        lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, _ = cv2.split(lab)

        # 2. HSV 색공간 변환
        hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        h, s, _ = cv2.split(hsv)

        # 붉은색 검출 조건 (조명에 강인한 복합 필터):
        # - CIE-Lab a* 채널: 128이 중립, 140 이상일수록 선명한 붉은색
        # - HSV Hue: 0~15 및 165~180 (원형 색상환 양 끝단)
        # - HSV Saturation: 60 이상 (채도가 있는 유효 착색)
        red_lab = a_channel > 138
        red_hsv1 = (h <= 15) & (s >= 60)
        red_hsv2 = (h >= 165) & (s >= 60)
        red_hsv = red_hsv1 | red_hsv2

        # Lab의 붉은 기운과 HSV의 색상 일치 영역 합성
        is_red = (red_lab | red_hsv) & (valid_mask > 0)

        red_pixels = np.count_nonzero(is_red)
        coverage_ratio = (red_pixels / total_pixels) * 100.0

        # 유효 사과 영역 내의 평균 a* 값 (적색 농도)
        mean_a = float(np.mean(a_channel[valid_mask > 0]))

        red_mask_vis = np.zeros_like(apple_mask)
        red_mask_vis[is_red] = 255

        return round(coverage_ratio, 2), round(mean_a, 2), red_mask_vis

    def detect_defects(
        self,
        bgr_img: np.ndarray,
        apple_mask: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        사과 표면의 멍, 흠집, 흑점 등 국소 결점 영역 비율 검출
        구형 자연 음영 오인식을 방지하기 위해 외곽 경계 침식 및 결점 면적 필터 적용
        """
        # 가장자리 음영 오검출 방지를 위해 마스크 내측 8% 침식
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        inner_mask = cv2.erode(apple_mask, kernel_erode, iterations=2)
        valid_mask = self.filter_specular_highlights(bgr_img, inner_mask)
        total_pixels = np.count_nonzero(valid_mask)
        if total_pixels == 0:
            return 0.0, np.zeros_like(apple_mask)

        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        
        # Black Top-Hat: 국소적인 어두운 얼룩(멍/상처) 추출
        kernel_hat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_hat)

        # 멍 및 흠집 임계값 적용
        _, defect_bin = cv2.threshold(blackhat, 40, 255, cv2.THRESH_BINARY)
        candidate_defects = cv2.bitwise_and(defect_bin, defect_bin, mask=valid_mask)

        # 유의미한 크기의 결점(면적 25픽셀 이상)만 필터링
        contours, _ = cv2.findContours(candidate_defects, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        defect_mask = np.zeros_like(apple_mask)
        valid_defect_pixels = 0

        for c in contours:
            area = cv2.contourArea(c)
            # 미세 노이즈 제외 및 실제 멍/결점 크기 필터
            if area >= 20:
                cv2.drawContours(defect_mask, [c], -1, 255, thickness=-1)
                valid_defect_pixels += int(area)

        defect_ratio = (valid_defect_pixels / total_pixels) * 100.0
        return round(defect_ratio, 2), defect_mask

    def inspect(
        self,
        bgr_img: np.ndarray,
        apple_mask: Optional[np.ndarray] = None
    ) -> QualityResult:
        """
        단일 이미지 외관 품질 종합 평가
        """
        if apple_mask is None:
            apple_mask = self.create_apple_mask(bgr_img)

        if np.count_nonzero(apple_mask) < 200:
            return QualityResult(
                grade_2_tier="미검출",
                grade_3_tier="미검출",
                color_ratio=0.0,
                defect_ratio=0.0,
                mean_a_star=0.0,
                is_valid=False,
                details={}
            )

        color_ratio, mean_a, _ = self.analyze_color_coverage(bgr_img, apple_mask)
        defect_ratio, _ = self.detect_defects(bgr_img, apple_mask)

        # 1. 기본 미션 기준 2등급 분류 (상: 밝은 색 / 중: 어두운 색)
        # 상: 붉은 착색 비율 60% 이상 및 적색도(a*) 135 이상
        # 중: 착색 부족, 녹황색 혼재 또는 어두운 적색 사과
        if color_ratio >= 60.0 and mean_a >= 135.0:
            grade_2 = "상"
        else:
            grade_2 = "중"

        # 2. 확장 미션 기준 3등급 분류 (특 / 상 / 보통 / 리젝트)
        # 농산물 표준규격 준용 + 결점 비율 반영
        if defect_ratio > 8.0:
            grade_3 = "불합격(리젝트)"
        elif color_ratio >= self.min_red_premium and defect_ratio < 2.0:
            grade_3 = "특"
        elif color_ratio >= self.min_red_standard and defect_ratio < 5.0:
            grade_3 = "상"
        elif color_ratio >= self.min_red_commercial:
            grade_3 = "보통"
        else:
            grade_3 = "불합격(리젝트)"

        return QualityResult(
            grade_2_tier=grade_2,
            grade_3_tier=grade_3,
            color_ratio=color_ratio,
            defect_ratio=defect_ratio,
            mean_a_star=mean_a,
            is_valid=True,
            details={
                "min_red_premium": self.min_red_premium,
                "min_red_standard": self.min_red_standard,
                "min_red_commercial": self.min_red_commercial
            }
        )

    def inspect_multi_view(self, view_images: List[np.ndarray]) -> QualityResult:
        """
        로봇팔이 사과를 집고 회전시키며 촬영한 다면(Multi-view) 이미지 종합 검사
        - 가장 품질이 낮은 면에 패널티를 부여하는 보수적 안전 판정
        """
        if not view_images:
            raise ValueError("검사할 뷰 이미지가 비어 있습니다.")

        results = [self.inspect(img) for img in view_images if img is not None]
        valid_results = [r for r in results if r.is_valid]

        if not valid_results:
            return QualityResult(
                grade_2_tier="미검출",
                grade_3_tier="미검출",
                color_ratio=0.0,
                defect_ratio=0.0,
                mean_a_star=0.0,
                is_valid=False,
                details={"view_count": 0}
            )

        # 전체 면 평균 착색률 및 최소 착색률 종합
        avg_color = float(np.mean([r.color_ratio for r in valid_results]))
        min_color = float(np.min([r.color_ratio for r in valid_results]))
        max_defect = float(np.max([r.defect_ratio for r in valid_results]))
        avg_a = float(np.mean([r.mean_a_star for r in valid_results]))

        # 종합 착색률: 평균 70% + 최저면 30% 반영
        final_color_score = round(0.7 * avg_color + 0.3 * min_color, 2)

        # 등급 재산정
        if final_color_score >= 60.0 and avg_a >= 142.0 and max_defect < 5.0:
            grade_2 = "상"
        else:
            grade_2 = "중"

        if max_defect > 8.0:
            grade_3 = "불합격(리젝트)"
        elif final_color_score >= self.min_red_premium and max_defect < 2.0:
            grade_3 = "특"
        elif final_color_score >= self.min_red_standard and max_defect < 5.0:
            grade_3 = "상"
        elif final_color_score >= self.min_red_commercial:
            grade_3 = "보통"
        else:
            grade_3 = "불합격(리젝트)"

        return QualityResult(
            grade_2_tier=grade_2,
            grade_3_tier=grade_3,
            color_ratio=final_color_score,
            defect_ratio=max_defect,
            mean_a_star=round(avg_a, 2),
            is_valid=True,
            details={
                "multi_view_count": len(valid_results),
                "avg_color_ratio": round(avg_color, 2),
                "min_color_ratio": round(min_color, 2),
                "max_defect_ratio": round(max_defect, 2)
            }
        )
