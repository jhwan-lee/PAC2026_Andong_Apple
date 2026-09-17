"""
PAC 2026 안동청과 미션 - 사과 품질 판별 YOLO 학습 스크립트
================================================================================
작성자: 이재환 (YOLO 사과 학습 & 품질 구분 파트장)
역할:
1. 사과 외관 품질 3개 클래스 (grade_high: 상, grade_mid: 중, defect: 결점) 학습
2. 어두운 공장 환경 및 반사광에 강인하도록 특수 Data Augmentation 파라미터 적용
3. WeGo PIPER 실시간 제어 사이클 타임 단축을 위한 경량 모델 (YOLOv8n / YOLOv11n) 활용
"""

import os
import sys
import argparse
from pathlib import Path


def train_yolo(
    data_yaml: str = "apple_data_template.yaml",
    model_type: str = "yolo11n.pt",  # 또는 yolov8n.pt
    epochs: int = 80,
    imgsz: int = 640,
    batch: int = 16,
    project_dir: str = "runs/train_apple",
    experiment_name: str = "apple_quality_v1",
    device: str = "0"  # GPU 없을 경우 "cpu"
):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics 패키지가 설치되어 있지 않습니다. pip install ultralytics 를 실행하세요.")
        sys.exit(1)

    print("=" * 70)
    print("PAC 2026 안동청과 - YOLO 사과 품질 모델 학습 파이프라인")
    print("=" * 70)
    print(f"• 데이터 설정 파일 : {data_yaml}")
    print(f"• 베이스 모델     : {model_type}")
    print(f"• 에포크(Epochs)  : {epochs}")
    print(f"• 입력 이미지 크기: {imgsz}x{imgsz}")
    print(f"• 배치 사이즈     : {batch}")
    print(f"• 디바이스        : {device}")
    print("=" * 70)

    # 사전 학습된 가중치 로드
    model = YOLO(model_type)

    # 공장 저조도 및 반사광 특화 하이퍼파라미터
    # - hsv_v: 0.4 (밝기 변동에 강인)
    # - hsv_s: 0.5 (착색 채도 변동에 강인)
    # - degrees: 180.0 (사과가 어느 방향으로 놓여도 회전 불변)
    # - fliplr: 0.5, flipud: 0.5 (좌우/상하 대칭)
    # - mosaic: 1.0 (사과 크기 및 위치 다양화)
    train_results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project_dir,
        name=experiment_name,
        # 조명 환경 극복을 위한 Data Augmentation
        hsv_h=0.03,        # 색조 미세 변화 허용
        hsv_s=0.5,         # 채도 변화 적극 반영
        hsv_v=0.4,         # 명도(어두움/밝음) 대폭 증강
        degrees=180.0,     # 360도 회전 증강
        translate=0.1,     # 트레이 위치 이동 증강
        scale=0.2,         # 크기 변화
        shear=2.0,         # 전단 변형
        perspective=0.0005,# 원근 왜곡
        flipud=0.5,        # 상하 뒤집기
        fliplr=0.5,        # 좌우 뒤집기
        mosaic=1.0,        # 모자이크 합성
        save=True,
        save_period=10,
        plots=True,
        verbose=True
    )

    print("\n[SUCCESS] 학습 완료!")
    best_pt = Path(project_dir) / experiment_name / "weights" / "best.pt"
    print(f"• 최적 가중치 저장 위치: {best_pt}")

    # 검증 성능 출력
    metrics = model.val()
    print(f"• 최종 mAP50: {metrics.box.map50:.4f}, mAP50-95: {metrics.box.map:.4f}")
    return model, best_pt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PAC 2026 Apple YOLO Trainer")
    parser.add_argument("--data", type=str, default="apple_data_template.yaml", help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Base model (yolov8n.pt, yolo11n.pt, etc.)")
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="0", help="CUDA device ('0') or 'cpu'")

    args = parser.parse_args()
    train_yolo(
        data_yaml=args.data,
        model_type=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device
    )
