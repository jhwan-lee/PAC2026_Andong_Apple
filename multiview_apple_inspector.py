"""
PAC 2026 안동청과 미션 - 360도 사과 다면(Multi-view) 품질 판정 및 로봇 분류 모듈
================================================================================
작성자: 이재환 & 아티 (A-ti)
목적:
- 사과를 360도 회전(손 회전 또는 로봇팔 손목 J6 회전)시키며 전 둘레의 품질을 다면 분석
- 단일 뷰에서 놓칠 수 있는 뒤편의 결점(멍/흠집) 및 착색 불량(Worst Face) 전수 검사
- 농관원 사과 표준규격 및 다면 가중치 합성(평균 60% + 최저면 40%) 적용
- 로봇 제어팀(WeGo PIPER)에 전달할 최종 분류 목적지(Box A/B/Reject) 및 3D 좌표 JSON 생성
- 한글 나눔폰트 기반 고품질 실시간 HUD 및 결과 리포트 대시보드 렌더링
"""

import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
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

FONT_PATH = "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"


class MultiViewAppleInspector:
    STATE_IDLE = 0       # 사과 탐색 및 대기
    STATE_SCANNING = 1   # 360도 회전 프레임 수집 중
    STATE_RESULT = 2     # 종합 판정 결과 대시보드 표시

    def __init__(
        self,
        weights_path: str = "/home/jaehwan/Documents/PAC2026_Andong_Apple/runs/detect/runs/train_apple/apple_quality_v1-2/weights/best.pt",
        backup_coco_weights: str = "/home/jaehwan/Documents/PAC2026_Andong_Apple/yolo_quality/yolo11n.pt",
        conf_thresh: float = 0.20,
        scan_total_frames: int = 75,  # 30fps 기준 약 2.5초 회전 수집
        width: int = 1280,
        height: int = 720,
        fps: int = 30
    ):
        self.conf_thresh = conf_thresh
        self.scan_total_frames = scan_total_frames
        self.width = width
        self.height = height
        self.fps = fps
        self.state = self.STATE_IDLE

        # 1. 폰트 로드
        self.font_large = ImageFont.truetype(FONT_PATH, 24) if os.path.exists(FONT_PATH) else None
        self.font_mid = ImageFont.truetype(FONT_PATH, 18) if os.path.exists(FONT_PATH) else None
        self.font_small = ImageFont.truetype(FONT_PATH, 14) if os.path.exists(FONT_PATH) else None

        # 2. YOLO 및 물리 비전 로드
        print(f"📦 [1/3] YOLO 가중치 로드 중: {weights_path}")
        self.model = YOLO(weights_path) if os.path.exists(weights_path) else YOLO(backup_coco_weights)
        self.coco_model = YOLO(backup_coco_weights) if os.path.exists(backup_coco_weights) else None
        self.phys_detector = AppleQualityDetector() if HAVE_PHYS_DETECTOR else None

        # 3. RealSense 초기화
        print("📷 [2/3] Intel RealSense D435 초기화 중...")
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        self.align = rs.align(rs.stream.color)
        self.colorizer = rs.colorizer()

        self.profile = self.pipeline.start(self.config)
        self.depth_intrinsics = None
        print("   ✅ D435 스트리밍 시작 완료")

        # 스캔 버퍼
        self.scanned_frames: List[Dict] = []
        self.final_report: Optional[Dict] = None
        self.dashboard_img: Optional[np.ndarray] = None

    def draw_text_ko(self, img: np.ndarray, text: str, pos: Tuple[int, int], font_type="mid", color=(255, 255, 255)) -> np.ndarray:
        """한글 텍스트 깨짐 없이 렌더링"""
        font = self.font_mid if font_type == "mid" else (self.font_large if font_type == "large" else self.font_small)
        if font is None:
            cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            return img
        
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        # BGR -> RGB 색상 반전
        rgb_color = (color[2], color[1], color[0])
        draw.text(pos, text, font=font, fill=rgb_color)
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def get_robust_depth_mm(self, depth_frame: rs.depth_frame, cx: int, cy: int, patch_radius: int = 5) -> float:
        w, h = depth_frame.get_width(), depth_frame.get_height()
        samples = []
        for dy in range(-patch_radius, patch_radius + 1):
            ny = cy + dy
            if ny < 0 or ny >= h:
                continue
            for dx in range(-patch_radius, patch_radius + 1):
                nx = cx + dx
                if nx < 0 or nx >= w:
                    continue
                d = depth_frame.get_distance(nx, ny)
                if d > 0.18:
                    samples.append(d)
        if samples:
            return float(np.median(samples)) * 1000.0
        d = depth_frame.get_distance(cx, cy)
        return float(d) * 1000.0 if d > 0 else 0.0

    def deproject_to_3d(self, cx: int, cy: int, depth_mm: float) -> Tuple[float, float, float]:
        if self.depth_intrinsics is None or depth_mm <= 0:
            return (0.0, 0.0, 0.0)
        d_m = depth_mm / 1000.0
        p = rs.rs2_deproject_pixel_to_point(self.depth_intrinsics, [cx, cy], d_m)
        return (round(p[0] * 1000.0, 1), round(p[1] * 1000.0, 1), round(depth_mm, 1))

    def detect_single_frame(self, color_img: np.ndarray, depth_frame: rs.depth_frame) -> Optional[Dict]:
        """단일 프레임 사과 검출 및 다면 특징 분석"""
        yolo_res = self.model.predict(color_img, conf=self.conf_thresh, verbose=False)[0]
        boxes = yolo_res.boxes

        target_box = None
        cls_id = 1
        conf = 0.0

        if len(boxes) > 0:
            # 화면 중심에 가장 가까운 사과 선택
            img_cx, img_cy = self.width // 2, self.height // 2
            best_dist = 999999
            for b in boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                bcx, bcy = (x1 + x2) // 2, (y1 + y2) // 2
                dist = (bcx - img_cx)**2 + (bcy - img_cy)**2
                if dist < best_dist:
                    best_dist = dist
                    target_box = (x1, y1, x2, y2)
                    cls_id = int(b.cls[0].item())
                    conf = float(b.conf[0].item())
        elif self.coco_model is not None:
            coco_res = self.coco_model.predict(color_img, conf=self.conf_thresh, verbose=False)[0]
            for b in coco_res.boxes:
                cname = coco_res.names.get(int(b.cls[0]), "")
                if cname == "apple" or int(b.cls[0]) == 47:
                    target_box = tuple(map(int, b.xyxy[0].tolist()))
                    cls_id = 0
                    conf = float(b.conf[0].item())
                    break

        if target_box is None:
            return None

        x1, y1, x2, y2 = target_box
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        crop = color_img[max(0, y1):min(self.height, y2), max(0, x1):min(self.width, x2)].copy()

        # 깊이 및 3D 좌표
        depth_mm = self.get_robust_depth_mm(depth_frame, cx, cy)
        X, Y, Z = self.deproject_to_3d(cx, cy, depth_mm)

        # 물리적 비전 품질 검사
        color_ratio = 50.0
        defect_ratio = 0.0
        mean_a = 135.0
        if self.phys_detector and crop.size > 0:
            q = self.phys_detector.inspect(crop)
            color_ratio = q.color_ratio
            defect_ratio = q.defect_ratio
            mean_a = q.mean_a_star

        # 클래스 매핑: 0: grade_high, 1: grade_mid, 2: defect
        has_defect = (cls_id == 2) or (defect_ratio > 4.0)

        return {
            "timestamp": time.time(),
            "box": (x1, y1, x2, y2),
            "center": (cx, cy),
            "coord_3d": (X, Y, Z),
            "crop": crop,
            "cls_id": cls_id,
            "conf": conf,
            "color_ratio": color_ratio,
            "defect_ratio": defect_ratio,
            "mean_a_star": mean_a,
            "has_defect": has_defect
        }

    def aggregate_multiview(self) -> Dict:
        """360도 회전 프레임 종합 통계 및 최종 등급 판정"""
        if not self.scanned_frames:
            return {}

        n = len(self.scanned_frames)
        colors = [f["color_ratio"] for f in self.scanned_frames]
        defects = [f["has_defect"] for f in self.scanned_frames]
        coords = [f["coord_3d"] for f in self.scanned_frames if f["coord_3d"][2] > 0]

        avg_color = float(np.mean(colors))
        min_color = float(np.min(colors))
        max_color = float(np.max(colors))
        defect_count = int(np.sum(defects))

        # 다면 가중 착색도: 평균 60% + 최저면 40% (보수적 결함 방지)
        composite_color_score = round(0.6 * avg_color + 0.4 * min_color, 1)

        # 3D 중심 좌표 평균
        if coords:
            avg_x = round(float(np.median([c[0] for c in coords])), 1)
            avg_y = round(float(np.median([c[1] for c in coords])), 1)
            avg_z = round(float(np.median([c[2] for c in coords])), 1)
        else:
            avg_x, avg_y, avg_z = 0.0, 0.0, 0.0

        # 등급 판정 로직
        # 1. 결함(멍/흠집)이 전체 회전 중 10% 이상 프레임에서 포착되면 결함 등급
        has_critical_defect = defect_count >= max(2, int(n * 0.08))

        if has_critical_defect:
            final_grade = "불합격 (결점/리젝트)"
            tier_2 = "중 (결점)"
            box_target = "REJECT_BIN"
            color_grade = (0, 0, 255)  # 빨간색
        elif composite_color_score >= 70.0 and min_color >= 45.0:
            final_grade = "특 (Premium)"
            tier_2 = "상 (특상)"
            box_target = "BOX_A"
            color_grade = (0, 230, 0)  # 초록색
        elif composite_color_score >= 50.0 and min_color >= 30.0:
            final_grade = "상 (Standard)"
            tier_2 = "상 (표준)"
            box_target = "BOX_A"
            color_grade = (0, 215, 255)  # 노란색
        else:
            final_grade = "보통 (Commercial)"
            tier_2 = "중 (보통)"
            box_target = "BOX_B"
            color_grade = (0, 165, 255)  # 주황색

        # 대표 각도 4장 선정 (0°, 90°, 180°, 270°)
        step = max(1, n // 4)
        sample_crops = [self.scanned_frames[min(i * step, n - 1)]["crop"] for i in range(4)]

        # 최저 착색면(Worst)과 최고 착색면(Best)
        min_idx = int(np.argmin(colors))
        max_idx = int(np.argmax(colors))
        worst_crop = self.scanned_frames[min_idx]["crop"]
        best_crop = self.scanned_frames[max_idx]["crop"]

        report = {
            "timestamp": int(time.time()),
            "total_frames_collected": n,
            "composite_color_score": composite_color_score,
            "avg_color": round(avg_color, 1),
            "min_color": round(min_color, 1),
            "max_color": round(max_color, 1),
            "defect_detected": has_critical_defect,
            "defect_frames_count": defect_count,
            "final_grade": final_grade,
            "tier_2": tier_2,
            "sorting_target_box": box_target,
            "grade_bgr_color": color_grade,
            "target_3d_mm": [avg_x, avg_y, avg_z],
            "sample_crops": sample_crops,
            "worst_crop": worst_crop,
            "best_crop": best_crop
        }
        return report

    def render_result_dashboard(self, base_img: np.ndarray, report: Dict) -> np.ndarray:
        """결과 대시보드 렌더링"""
        board = base_img.copy()

        # 반투명 어두운 배경 오버레이 (우측 절반 패널: x=720 ~ 1260)
        panel_w = 540
        overlay = board.copy()
        cv2.rectangle(overlay, (self.width - panel_w - 20, 20), (self.width - 20, self.height - 20), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.88, board, 0.12, 0, board)
        cv2.rectangle(board, (self.width - panel_w - 20, 20), (self.width - 20, self.height - 20), report["grade_bgr_color"], 2)

        x_start = self.width - panel_w
        y = 55

        # 1. 헤더 타이틀
        board = self.draw_text_ko(board, "🍎 PAC 2026 사과 360도 품질 종합 성적서", (x_start, y), "large", (0, 255, 255))
        y += 45

        # 2. 최종 등급 하이라이트
        cv2.rectangle(board, (x_start - 5, y - 5), (self.width - 35, y + 55), (40, 40, 40), -1)
        cv2.rectangle(board, (x_start - 5, y - 5), (self.width - 35, y + 55), report["grade_bgr_color"], 2)
        grade_text = f"최종 등급: {report['final_grade']}"
        board = self.draw_text_ko(board, grade_text, (x_start + 10, y + 5), "large", report["grade_bgr_color"])
        y += 75

        # 3. 로봇 분류 목적지
        box_text = f"🤖 로봇 이송 목적지: [{report['sorting_target_box']}] (2등급: {report['tier_2']})"
        board = self.draw_text_ko(board, box_text, (x_start, y), "mid", (255, 255, 255))
        y += 35

        # 4. 3D 공간 좌표
        x, y_c, z = report["target_3d_mm"]
        if z > 0:
            coord_str = f"📍 3D 파지 좌표: X={x:+.1f}mm, Y={y_c:+.1f}mm, Z={z:.1f}mm"
        else:
            coord_str = "📍 3D 파지 좌표: Z < 20cm (초근접 데드존)"
        board = self.draw_text_ko(board, coord_str, (x_start, y), "mid", (0, 255, 200))
        y += 40

        # 5. 착색도 및 결함 통계
        stats_1 = f"• 종합 착색 점수: {report['composite_color_score']}%  (평균 60% + 최저 40%)"
        stats_2 = f"• 최저면(Worst): {report['min_color']}%  |  최고면(Best): {report['max_color']}%"
        defect_str = "• 멍/결점(Defect): 감지됨! (강등)" if report["defect_detected"] else "• 멍/결점(Defect): 없음 (정상)"
        def_color = (0, 0, 255) if report["defect_detected"] else (0, 255, 0)

        board = self.draw_text_ko(board, stats_1, (x_start, y), "mid", (220, 220, 220))
        y += 28
        board = self.draw_text_ko(board, stats_2, (x_start, y), "small", (180, 180, 180))
        y += 28
        board = self.draw_text_ko(board, defect_str, (x_start, y), "mid", def_color)
        y += 40

        # 6. 다면 썸네일 표시 (4개 각도 썸네일)
        board = self.draw_text_ko(board, "📸 360도 회전 다면 스캔 썸네일 (4개 방위)", (x_start, y), "mid", (255, 200, 100))
        y += 30

        thumb_w, thumb_h = 110, 110
        gap = 15
        for i, crop in enumerate(report["sample_crops"]):
            if crop.size > 0:
                resized = cv2.resize(crop, (thumb_w, thumb_h))
                bx = x_start + i * (thumb_w + gap)
                by = y
                if bx + thumb_w < self.width - 20:
                    board[by:by + thumb_h, bx:bx + thumb_w] = resized
                    cv2.rectangle(board, (bx, by), (bx + thumb_w, by + thumb_h), (100, 100, 100), 1)
                    angle_label = f"{i * 90}°"
                    cv2.putText(board, angle_label, (bx + 5, by + thumb_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        y += thumb_h + 30

        # 7. 조작 안내
        guide_text = "[R] 다음 사과 리셋 | [S] 결과 저장 | [Q] 종료"
        board = self.draw_text_ko(board, guide_text, (x_start + 40, y), "mid", (0, 255, 255))

        return board

    def run(self):
        print("=" * 65)
        print("🍎 [PAC 2026] 360도 사과 다면 품질 검사 시스템 준비 완료")
        print("   1. 사과를 카메라 앞 30~45cm 거리에 위치시킵니다.")
        print("   2. [스페이스바]를 누르고 사과를 천천히 한 바퀴 360도 돌려주세요.")
        print("   3. 회전이 끝나면 종합 품질 성적서와 로봇 분류 명령이 출력됩니다.")
        print("   4. 'q' 키 또는 ESC: 종료")
        print("=" * 65)

        save_report_path = "multiview_inspection_result.png"

        try:
            while True:
                frames = self.pipeline.wait_for_frames(timeout_ms=5000)
                aligned = self.align.process(frames)
                depth_f = aligned.get_depth_frame()
                color_f = aligned.get_color_frame()
                if not depth_f or not color_f:
                    continue

                if self.depth_intrinsics is None:
                    self.depth_intrinsics = depth_f.profile.as_video_stream_profile().intrinsics

                color_img = np.asanyarray(color_f.get_data())
                depth_colormap = np.asanyarray(self.colorizer.colorize(depth_f).get_data())

                # 단일 프레임 사과 검출
                item = self.detect_single_frame(color_img, depth_f)
                display_img = color_img.copy()

                # === 상태 머신 처리 ===
                if self.state == self.STATE_IDLE:
                    # 대기 상태: 현재 보이는 사과와 조작 안내문 표시
                    if item:
                        x1, y1, x2, y2 = item["box"]
                        cx, cy = item["center"]
                        cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        cv2.drawMarker(display_img, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
                        label = f"사과 감지됨 (착색: {item['color_ratio']:.0f}%)"
                        display_img = self.draw_text_ko(display_img, label, (x1, max(20, y1 - 25)), "mid", (0, 255, 255))

                    # 안내 배너
                    banner_y = 35
                    cv2.rectangle(display_img, (0, 0), (self.width, 50), (15, 15, 15), -1)
                    guide = "👉 [스페이스바]를 누르고 사과를 천천히 한 바퀴(360°) 돌려주세요!"
                    display_img = self.draw_text_ko(display_img, guide, (30, 12), "large", (0, 255, 200))

                elif self.state == self.STATE_SCANNING:
                    # 회전 검사 수집 중
                    if item:
                        self.scanned_frames.append(item)
                        x1, y1, x2, y2 = item["box"]
                        cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 3)

                    collected_n = len(self.scanned_frames)
                    progress = min(1.0, collected_n / self.scan_total_frames)

                    # 상단 프로그레스 바
                    cv2.rectangle(display_img, (0, 0), (self.width, 60), (20, 20, 20), -1)
                    bar_w = int((self.width - 60) * progress)
                    cv2.rectangle(display_img, (30, 42), (30 + bar_w, 54), (0, 230, 0), -1)
                    cv2.rectangle(display_img, (30, 42), (self.width - 30, 54), (100, 100, 100), 1)

                    scan_msg = f"🔄 360도 회전 검사 진행 중... [{collected_n} / {self.scan_total_frames} 프레임] ({progress*100:.0f}%)"
                    display_img = self.draw_text_ko(display_img, scan_msg, (30, 10), "large", (0, 255, 255))

                    # 수집 완료 조건 (설정 프레임 도달)
                    if collected_n >= self.scan_total_frames:
                        print("✨ [360도 스캔 완료] 다면 품질 집계 및 로봇 패킷 생성 중...")
                        self.final_report = self.aggregate_multiview()
                        self.dashboard_img = self.render_result_dashboard(color_img, self.final_report)
                        cv2.imwrite(save_report_path, self.dashboard_img)
                        print(f"📄 종합 성적서 저장 완료: {save_report_path}")

                        # 로봇 연동 JSON 패킷 터미널 출력
                        robot_packet = {
                            "header": {"sender": "VISION_MULTIVIEW", "timestamp": self.final_report["timestamp"]},
                            "decision": {
                                "final_grade": self.final_report["final_grade"],
                                "tier_2": self.final_report["tier_2"],
                                "target_box": self.final_report["sorting_target_box"],
                                "composite_score": self.final_report["composite_color_score"],
                                "min_face_color": self.final_report["min_color"],
                                "defect_detected": self.final_report["defect_detected"],
                                "pick_xyz_mm": self.final_report["target_3d_mm"]
                            }
                        }
                        print("\n" + "="*50)
                        print("🤖 [WeGo PIPER 로봇팔 전송용 JSON 패킷]")
                        print(json.dumps(robot_packet, indent=2, ensure_ascii=False))
                        print("="*50 + "\n")

                        self.state = self.STATE_RESULT

                elif self.state == self.STATE_RESULT:
                    # 결과 대시보드 고정 표시
                    if self.dashboard_img is not None:
                        display_img = self.dashboard_img

                # 우측 하단 미니 뎁스맵
                if self.state != self.STATE_RESULT:
                    mini_d = cv2.resize(depth_colormap, (240, 135))
                    display_img[self.height - 145:self.height - 10, self.width - 250:self.width - 10] = mini_d
                    cv2.rectangle(display_img, (self.width - 250, self.height - 145), (self.width - 10, self.height - 10), (0, 255, 200), 1)

                cv2.imshow("PAC 2026 Apple 360 Quality Inspector", display_img)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == 27:
                    break
                elif key == 32:  # SPACE BAR
                    if self.state == self.STATE_IDLE:
                        print("▶️ [회전 검사 시작] 사과를 천천히 돌려주세요...")
                        self.scanned_frames.clear()
                        self.state = self.STATE_SCANNING
                    elif self.state == self.STATE_SCANNING:
                        # 조기 완료
                        if len(self.scanned_frames) >= 10:
                            print("⏹️ [조기 완료] 다면 품질 집계 중...")
                            self.final_report = self.aggregate_multiview()
                            self.dashboard_img = self.render_result_dashboard(color_img, self.final_report)
                            cv2.imwrite(save_report_path, self.dashboard_img)
                            self.state = self.STATE_RESULT
                elif key == ord('r'):
                    print("🔄 [리셋] 대기 모드로 전환합니다.")
                    self.scanned_frames.clear()
                    self.final_report = None
                    self.dashboard_img = None
                    self.state = self.STATE_IDLE
                elif key == ord('s'):
                    snap_name = f"snapshot_360_{int(time.time())}.png"
                    cv2.imwrite(snap_name, display_img)
                    print(f"📸 스냅샷 저장: {snap_name}")

        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()
            print("🛑 360도 검사 모듈 안전 종료 완료.")


def main():
    inspector = MultiViewAppleInspector()
    inspector.run()


if __name__ == "__main__":
    main()
