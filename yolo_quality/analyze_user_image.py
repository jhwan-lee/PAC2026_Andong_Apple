import os
import sys
import cv2
import numpy as np

# 경로 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "yolo_quality"))

from infer_apple_quality import AppleYOLOClassifier
from apple_quality_detector import AppleQualityDetector

img_path = "/home/jaehwan/.gemini/antigravity/brain/1d8d602d-c61c-4107-9043-e662f790ad05/.user_uploaded/media_1789654366625.png"

if not os.path.exists(img_path):
    print("Image not found:", img_path)
    sys.exit(1)

img = cv2.imread(img_path)
h, w = img.shape[:2]
print(f"이미지 크기: {w}x{h}")

# 1. YOLO 분류기 추론
classifier = AppleYOLOClassifier(conf_threshold=0.3)
detections = classifier.predict_tray(img)

print(f"\n[YOLO 검출 결과] 총 {len(detections)}개 사과 검출")
for d in detections:
    print(f"ID {d.target_id}:")
    print(f"  • 클래스: {d.class_name} ({d.class_id}) -> 등급: [{d.grade_korean}]")
    print(f"  • 신뢰도: {d.confidence:.2%}")
    print(f"  • 바운딩 박스: {d.box_xyxy}")
    print(f"  • 중심점: {d.center_px}, 반경: {d.radius_px}px")
    print(f"  • 물리 착색률: {d.color_ratio_phys:.1f}%")

# 2. 물리 비전 정밀 검출기 단독 분석 (상세 리포트용)
detector = AppleQualityDetector()
# 만약 YOLO 검출 결과가 있으면 해당 박스 크롭, 없으면 전체 이미지
if detections:
    x1, y1, x2, y2 = detections[0].box_xyxy
    crop = img[y1:y2, x1:x2]
else:
    crop = img

quality_res = detector.inspect(crop)
print(f"\n[물리 비전 정밀 분석 리포트]")
print(f"  • 기본 2등급: [{quality_res.grade_2_tier}]")
print(f"  • 확장 3등급: [{quality_res.grade_3_tier}]")
print(f"  • 유효 착색 비율: {quality_res.color_ratio:.2f}%")
print(f"  • 평균 CIE-Lab a* 적색도: {quality_res.mean_a_star:.2f}")
print(f"  • 표면 결점(멍/흠집) 비율: {quality_res.defect_ratio:.2f}%")

# 3. 종합 시각화 대시보드 이미지 생성
color_ratio, mean_a, red_mask = detector.analyze_color_coverage(crop)
defect_ratio, defect_mask = detector.detect_defects(crop, detector.create_apple_mask(crop))

# 결과 시각화 합성
vis_main = img.copy()
if detections:
    vis_main = classifier.draw_detections(vis_main, detections)
else:
    # 수동 바운딩 박스
    pass

# 착색 히트맵 오버레이
crop_overlay = crop.copy()
red_overlay = np.zeros_like(crop)
red_overlay[red_mask > 0] = [0, 0, 255] # 빨간색
crop_colored = cv2.addWeighted(crop_overlay, 0.65, red_overlay, 0.35, 0)

out_dir = "/home/jaehwan/Documents/PAC2026_Andong_Apple/yolo_quality"
out_img_path = os.path.join(out_dir, "user_apple_analysis_result.png")

# 대시보드 구성: 왼쪽(YOLO 검출 및 바운딩박스), 오른쪽(착색 마스크 및 결점 분석)
target_h = 700
scale_main = target_h / h
w_main = int(w * scale_main)
vis_resized = cv2.resize(vis_main, (w_main, target_h))

crop_h, crop_w = crop.shape[:2]
scale_crop = (target_h // 2) / crop_h
w_crop = int(crop_w * scale_crop)
crop_vis = cv2.resize(crop_colored, (w_crop, target_h // 2))

# 결점 마스크 시각화
defect_vis_crop = crop.copy()
defect_vis_crop[defect_mask > 0] = [0, 255, 255] # 노란색 결점 표시
defect_vis = cv2.resize(defect_vis_crop, (w_crop, target_h // 2))

right_panel = np.vstack([crop_vis, defect_vis])

# 텍스트 라벨 추가
cv2.putText(right_panel, "Color Coverage Map (Red)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
cv2.putText(right_panel, f"Coverage: {quality_res.color_ratio:.1f}%", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

cv2.putText(right_panel, "Surface Defect Map", (15, (target_h // 2) + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
cv2.putText(right_panel, f"Defect: {quality_res.defect_ratio:.1f}%", (15, (target_h // 2) + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

dashboard = np.hstack([vis_resized, right_panel])
cv2.imwrite(out_img_path, dashboard)
print(f"\n[SAVE] 분석 대시보드 저장 완료: {out_img_path}")

# 아티팩트 디렉토리에도 복사
artifact_dir = "/home/jaehwan/.gemini/antigravity/brain/1d8d602d-c61c-4107-9043-e662f790ad05"
shutil_copy = f"cp '{out_img_path}' '{artifact_dir}/user_apple_analysis_result.png'"
os.system(shutil_copy)
print("Artifact copy finished!")
