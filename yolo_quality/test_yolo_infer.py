import os
import sys
import cv2

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# yolo_quality 모듈 import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from infer_apple_quality import AppleYOLOClassifier

def test():
    clf = AppleYOLOClassifier()
    img_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_tray_apples.png")
    if not os.path.exists(img_path):
        print(f"이미지 파일 없음: {img_path}")
        return

    img = cv2.imread(img_path)
    results = clf.predict_tray(img)
    print(f"\n[추론 결과] 검출 사과 수: {len(results)}개")
    for r in results:
        print(f"ID {r.target_id}: 등급=[{r.grade_korean}], 중심={r.center_px}, 반경={r.radius_px}px, 물리착색률={r.color_ratio_phys:.1f}%")

    vis = clf.draw_detections(img, results)
    out_path = "yolo_test_result.png"
    cv2.imwrite(out_path, vis)
    print(f"시각화 결과 저장: {out_path}")

if __name__ == "__main__":
    test()
