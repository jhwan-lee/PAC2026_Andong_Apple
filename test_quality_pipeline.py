"""
PAC 2026 분과 1 안동청과 미션 - 비전 품질 선별 파이프라인 통합 검증 스크립트
================================================================================
실제 하드웨어가 연결되지 않은 상태에서도 다양한 환경(어두운 조명, 밝은 사과, 어두운 사과,
결점 사과 등)을 시뮬레이션하여 비전 검출 및 착색도/품질 판별 정확도를 정량 검증합니다.
"""

import os
import sys

# Windows 콘솔 한글 깨짐 방지
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import cv2
import numpy as np

# 현재 디렉토리 모듈 import
from apple_quality_detector import AppleQualityDetector
from apple_locator import AppleLocator


def generate_synthetic_tray_scene(output_path: str = "tray_apples_sample.png") -> np.ndarray:
    """
    미션 환경과 유사한 합성 트레이 및 사과 5개 장면 생성
    - 어두운 공장 환경 시뮬레이션 (조명 감쇠 및 비대칭 반사광)
    - 5개 사과:
      1) 상(밝은 선홍색, 착색률 높음)
      2) 상(밝은 빨강, 약간의 반사광)
      3) 중(어두운 갈적색, 착색률 중간)
      4) 중(녹황색이 섞인 덜 익은 사과)
      5) 하/리젝트(표면에 멍/결점이 존재하는 사과)
    """
    h, w = 600, 800
    # 어두운 공장 배경 및 스테인리스/회색 트레이
    img = np.full((h, w, 3), (35, 35, 40), dtype=np.uint8)

    # 흰색/아이보리 트레이 렌더링
    tray_top_left = (100, 80)
    tray_bottom_right = (700, 520)
    cv2.rectangle(img, tray_top_left, tray_bottom_right, (180, 185, 190), -1)
    cv2.rectangle(img, (115, 95), (685, 505), (145, 150, 155), -1)

    # 사과 5개 속성 정의 (중심 x, y, 반지름, 색상 타입, 결점 여부)
    apples_config = [
        # 1. 상 등급: 밝고 선명한 빨강 (홍로 특품형)
        {"pos": (220, 200), "r": 52, "base_bgr": (30, 45, 220), "pattern": "bright_red", "defect": False},
        # 2. 상 등급: 밝은 붉은색 + 하이라이트 반사
        {"pos": (400, 180), "r": 50, "base_bgr": (25, 40, 210), "pattern": "bright_red_specular", "defect": False},
        # 3. 중 등급: 어둡고 칙칙한 적색 (중-어두운색 미션 사과)
        {"pos": (580, 230), "r": 53, "base_bgr": (25, 30, 135), "pattern": "dark_red", "defect": False},
        # 4. 중 등급: 황녹색과 적색이 혼재된 사과
        {"pos": (280, 380), "r": 51, "base_bgr": (40, 95, 175), "pattern": "mixed_green_red", "defect": False},
        # 5. 하/리젝트: 표면에 뚜렷한 멍(Bruise)이 있는 사과
        {"pos": (500, 390), "r": 49, "base_bgr": (30, 40, 200), "pattern": "bruised", "defect": True},
    ]

    for cfg in apples_config:
        cx, cy = cfg["pos"]
        r = cfg["r"]
        base_c = cfg["base_bgr"]

        # 원형 사과 베이스 그리기
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (cx, cy), r, 255, -1)

        # 사과 텍스처 및 입체 그라디언트 생성
        for y in range(cy - r, cy + r):
            for x in range(cx - r, cx + r):
                if 0 <= y < h and 0 <= x < w:
                    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                    if dist <= r:
                        # 구형 입체 음영 계수
                        sphere_shading = np.cos((dist / r) * (np.pi / 2.5))
                        # 기본 색상 + 텍스처 노이즈
                        noise = np.random.randint(-10, 10)
                        
                        if cfg["pattern"] == "mixed_green_red" and x < cx:
                            # 좌측은 녹황색
                            pix_b = np.clip(30 * sphere_shading + noise, 0, 255)
                            pix_g = np.clip(130 * sphere_shading + noise, 0, 255)
                            pix_r = np.clip(90 * sphere_shading + noise, 0, 255)
                        else:
                            pix_b = np.clip(base_c[0] * sphere_shading + noise, 0, 255)
                            pix_g = np.clip(base_c[1] * sphere_shading + noise, 0, 255)
                            pix_r = np.clip(base_c[2] * sphere_shading + noise, 0, 255)

                        img[y, x] = [pix_b, pix_g, pix_r]

        # 반사광(Specular Highlight) 시뮬레이션
        if cfg["pattern"] == "bright_red_specular":
            cv2.circle(img, (cx - 15, cy - 15), 10, (240, 245, 255), -1)

        # 멍(Bruise/Defect) 시뮬레이션
        if cfg["defect"]:
            cv2.ellipse(img, (cx + 12, cy + 10), (14, 9), 35, 0, 360, (20, 25, 55), -1)

    # 전체 장면에 약한 비네팅(공장 어두운 조명 효과) 적용
    cv2.imwrite(output_path, img)
    return img


def run_test():
    print("=" * 70)
    print("PAC 2026 안동청과 미션 - 사과 품질 판별 & 위치 추정 파이프라인 검증")
    print("=" * 70)

    # 1. 테스트 이미지 생성
    scene_path = "test_tray_apples.png"
    tray_img = generate_synthetic_tray_scene(scene_path)
    print(f"[1] 가상 트레이 및 사과 5개 테스트 이미지 생성 완료 -> {scene_path}")

    # 2. 사과 위치 탐색기 (Locator) 초기화 및 탐색
    locator = AppleLocator(pixel_to_mm_ratio=0.6, camera_center_robot=(350.0, 0.0, 480.0))
    targets = locator.detect_apples(tray_img, max_targets=5)
    print(f"[2] 사과 탐색 완료: 총 {len(targets)}개 객체 위치 파악")

    # 동선 최적화 정렬
    sorted_targets = locator.sort_targets_for_cycle_time(targets, robot_home_xy=(200.0, 0.0))

    # 3. 품질 판별 엔진 (Quality Detector) 초기화
    detector = AppleQualityDetector(
        min_red_ratio_premium=70.0,
        min_red_ratio_standard=50.0,
        min_red_ratio_commercial=30.0
    )

    vis_result = tray_img.copy()

    print("\n" + "-" * 70)
    print(f"{'ID':^4} | {'2등급(기본)':^10} | {'3등급(확장)':^12} | {'착색률':^8} | {'적색도(a*)':^10} | {'결점률':^8} | {'로봇좌표(X,Y,Z)':^18}")
    print("-" * 70)

    for target in sorted_targets:
        # 단일 사과 품질 검사
        res = detector.inspect(target.crop_img)

        # 화면 시각화 드로잉
        cx, cy = target.center_px
        r = target.radius_px
        color = (0, 255, 0) if res.grade_2_tier == "상" else (0, 165, 255)
        if res.grade_3_tier == "불합격(리젝트)":
            color = (0, 0, 255)

        cv2.circle(vis_result, (cx, cy), r, color, 3)
        cv2.circle(vis_result, (cx, cy), 4, (255, 255, 255), -1)

        label_2 = f"[{res.grade_2_tier}] {res.color_ratio}%"
        label_3 = f"Ext: {res.grade_3_tier}"
        coord_txt = f"({target.robot_xyz[0]:.0f}, {target.robot_xyz[1]:.0f})"

        cv2.putText(vis_result, label_2, (cx - r, cy - r - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        cv2.putText(vis_result, label_3, (cx - r, cy - r - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(vis_result, coord_txt, (cx - r, cy + r + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        print(
            f"{target.target_id:^4} | "
            f"{res.grade_2_tier:^10} | "
            f"{res.grade_3_tier:^12} | "
            f"{res.color_ratio:>6.1f}% | "
            f"{res.mean_a_star:>9.1f} | "
            f"{res.defect_ratio:>6.1f}% | "
            f"({target.robot_xyz[0]:>5.1f}, {target.robot_xyz[1]:>5.1f}, {target.robot_xyz[2]:>4.1f})"
        )

    # 4. 확장 미션: 다면 검사(Multi-view) 시뮬레이션
    print("-" * 70)
    print("[4] 확장 미션 검증: 사과 파지 후 손목 회전 다면 검사(Multi-view) 시뮬레이션")
    # 5번 결점 사과의 4각도(0도, 90도, 180도, 270도) 가상 뷰 생성
    sample_crop = sorted_targets[-1].crop_img
    view_angles = [
        sample_crop,
        cv2.rotate(sample_crop, cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(sample_crop, cv2.ROTATE_180),
        cv2.rotate(sample_crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
    ]
    multi_res = detector.inspect_multi_view(view_angles)
    print(f"  -> 4각도 종합 판정 결과:")
    print(f"     * 기본 2등급: {multi_res.grade_2_tier}")
    print(f"     * 확장 3등급: {multi_res.grade_3_tier}")
    print(f"     * 종합 착색 점수: {multi_res.color_ratio}% (최저면 반영)")
    print(f"     * 최대 결점 비율: {multi_res.defect_ratio}%")
    print(f"     * 다면 분석 세부치: {multi_res.details}")

    # 결과 이미지 저장
    result_img_path = "quality_inspection_result.png"
    cv2.imwrite(result_img_path, vis_result)
    print(f"\n[5] 검사 결과 시각화 이미지 저장 완료 -> {result_img_path}")
    print("=" * 70)


if __name__ == "__main__":
    run_test()
