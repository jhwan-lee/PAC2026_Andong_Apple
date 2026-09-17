"""
PAC 2026 안동청과 미션 - 실전 사과 데이터셋 자동 통합 및 변환 빌더
================================================================================
역할:
1. apples-fvpl5 (697장 COCO 포맷 사과) + rotten_apples (150장 결점 사과) 통합
2. 물리 비전 엔진(CIE-Lab+HSV)을 결합한 스마트 의사 라벨링:
   - 착색률 60% 이상: Class 0 (grade_high, 상 등급)
   - 착색률 60% 미만: Class 1 (grade_mid, 중 등급)
   - 부패/결점 사과: Class 2 (defect, 결점/리젝트)
3. YOLO 표준 포맷 (apple_dataset/images/{train,val,test}, labels/{train,val,test}) 생성
"""

import os
import sys
import json
import shutil
import cv2
from pathlib import Path
from tqdm import tqdm

# 상위 폴더의 물리 비전 검출기 import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from apple_quality_detector import AppleQualityDetector

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw_datasets"
TARGET_DIR = BASE_DIR / "apple_dataset"

def init_target_dirs():
    for split in ["train", "val", "test"]:
        (TARGET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (TARGET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

def process_rotten_apples():
    print("\n[1/2] 결점 사과(Rotten Apples) 데이터셋 변환 중...")
    rotten_root = RAW_DIR / "rotten_apples" / "RottenApples.v1i.yolov5pytorch (1)"
    if not rotten_root.exists():
        print(f"  [SKIP] {rotten_root} 경로가 없습니다.")
        return 0

    split_map = {"train": "train", "valid": "val", "test": "test"}
    count = 0

    for raw_split, target_split in split_map.items():
        img_dir = rotten_root / raw_split / "images"
        lbl_dir = rotten_root / raw_split / "labels"
        if not img_dir.exists():
            continue

        for img_path in img_dir.glob("*.jpg"):
            stem = img_path.stem
            lbl_path = lbl_dir / f"{stem}.txt"

            target_img = TARGET_DIR / "images" / target_split / f"rotten_{stem}.jpg"
            target_lbl = TARGET_DIR / "labels" / target_split / f"rotten_{stem}.txt"

            shutil.copy2(img_path, target_img)

            # 라벨 변환: 기존 0번(rotten) -> 우리 클래스 2번(defect)
            if lbl_path.exists():
                with open(lbl_path, "r") as f:
                    lines = f.readlines()
                new_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if parts:
                        parts[0] = "2" # defect 클래스
                        new_lines.append(" ".join(parts))
                with open(target_lbl, "w") as f:
                    f.write("\n".join(new_lines) + "\n")
            count += 1

    print(f"  -> 결점 사과 총 {count}장 변환 완료 (Class 2: defect)")
    return count

def process_apples_fvpl5():
    print("\n[2/2] 고해상도 사과 데이터셋(apples-fvpl5) 물리 착색도 스마트 라벨링 중...")
    rf_root = RAW_DIR / "home/zuppif/Documents/Work/RoboFlow/ODinW-RF100-challenge/rf100/apples-fvpl5"
    if not rf_root.exists():
        print(f"  [SKIP] {rf_root} 경로가 없습니다.")
        return 0

    detector = AppleQualityDetector()
    split_map = {"train": "train", "valid": "val", "test": "test"}
    stats = {0: 0, 1: 0, 2: 0}
    total_count = 0

    for raw_split, target_split in split_map.items():
        split_dir = rf_root / raw_split
        ann_file = split_dir / "_annotations.coco.json"
        if not ann_file.exists():
            continue

        with open(ann_file, "r") as f:
            coco = json.load(f)

        img_map = {img["id"]: img for img in coco["images"]}
        ann_map = {}
        for ann in coco["annotations"]:
            img_id = ann["image_id"]
            ann_map.setdefault(img_id, []).append(ann)

        print(f"  * {raw_split} ({len(img_map)}장) 분석 및 라벨 생성...")
        for img_id, img_info in tqdm(img_map.items(), desc=raw_split):
            file_name = img_info["file_name"]
            src_img_p = split_dir / file_name
            if not src_img_p.exists():
                continue

            img = cv2.imread(str(src_img_p))
            if img is None:
                continue

            h, w = img.shape[:2]
            target_img_p = TARGET_DIR / "images" / target_split / f"norm_{file_name}"
            target_lbl_p = TARGET_DIR / "labels" / target_split / f"norm_{Path(file_name).stem}.txt"

            shutil.copy2(src_img_p, target_img_p)

            yolo_lines = []
            for ann in ann_map.get(img_id, []):
                bx, by, bw, bh = ann["bbox"]
                if bw <= 0 or bh <= 0:
                    continue

                # 사과 영역 크롭
                x1, y1 = max(0, int(bx)), max(0, int(by))
                x2, y2 = min(w, int(bx + bw)), min(h, int(by + bh))
                crop = img[y1:y2, x1:x2]

                # 물리 비전 엔진으로 착색률 및 결점 분석
                cls_id = 0 # 기본: grade_high
                if crop.size > 0:
                    res = detector.inspect(crop)
                    if res.defect_ratio >= 6.0:
                        cls_id = 2 # 결점
                    elif res.color_ratio < 60.0:
                        cls_id = 1 # 중 등급 (착색 부족 또는 미숙)
                    else:
                        cls_id = 0 # 상 등급 (고착색)
                
                stats[cls_id] += 1

                # YOLO 포맷 정규화: class cx cy nw nh
                cx = (bx + bw / 2.0) / w
                cy = (by + bh / 2.0) / h
                nw = bw / w
                nh = bh / h
                yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            with open(target_lbl_p, "w") as f:
                f.write("\n".join(yolo_lines) + "\n")
            total_count += 1

    print(f"  -> 사과 데이터셋 총 {total_count}장 라벨링 완료!")
    print(f"     [클래스 분포] 상(grade_high): {stats[0]}개, 중(grade_mid): {stats[1]}개, 결점(defect): {stats[2]}개")
    return total_count

def update_yaml():
    yaml_content = f"""# PAC 2026 안동청과 미션 - 사과 품질 선별 YOLO 데이터셋
path: {TARGET_DIR.resolve()}
train: images/train
val: images/val
test: images/test

names:
  0: grade_high
  1: grade_mid
  2: defect
"""
    yaml_p = BASE_DIR / "apple_data.yaml"
    with open(yaml_p, "w") as f:
        f.write(yaml_content)
    print(f"\n✅ 데이터셋 설정 파일 생성 완료: {yaml_p}")

if __name__ == "__main__":
    init_target_dirs()
    c1 = process_rotten_apples()
    c2 = process_apples_fvpl5()
    update_yaml()
    print("\n🎉 모든 레퍼런스 데이터 통합 및 YOLO 변환이 완료되었습니다!")
