#!/bin/bash
# ==============================================================================
# PAC 2026 안동청과 미션 - 우분투 원클릭 자동 실행 스크립트
# 실행 명령어: ./run_pac.sh 또는 단축 명령어 'pac'
# ==============================================================================

# 스크립트가 위치한 디렉토리로 이동
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

echo "======================================================================"
echo "🍎 PAC 2026 안동청과 미션 - 자동 실행 시스템 (Powered by 아티)"
echo "   작업 디렉토리: $SCRIPT_DIR"
echo "======================================================================"

# 1. 파이썬 가상환경 점검 및 생성
if [ ! -d ".venv" ]; then
    echo "[1/4] 파이썬 가상환경(.venv)이 없습니다. 가상환경을 생성합니다..."
    python3 -m venv .venv
    if [ $? -ne 0 ]; then
        echo "❌ [오류] python3-venv 패키지가 필요합니다. sudo apt install python3-venv 를 실행하세요."
        exit 1
    fi
fi

# 가상환경 활성화
source .venv/bin/activate
echo "✅ [2/4] 가상환경 활성화 완료: $(which python3)"

# 2. 필수 라이브러리 설치 확인 (ultralytics, opencv, torch 등)
python3 -c "import ultralytics, cv2, torch" &> /dev/null
if [ $? -ne 0 ]; then
    echo "[3/4] 필수 패키지가 설치되지 않았거나 누락되었습니다. requirements.txt 설치를 진행합니다..."
    pip install --upgrade pip
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ [오류] 패키지 설치 중 오류가 발생했습니다. 네트워크 상태를 확인하세요."
        exit 1
    fi
else
    echo "✅ [3/4] 필수 패키지 (PyTorch, Ultralytics, OpenCV) 점검 완료!"
fi

echo "======================================================================"
echo "🎯 실행할 작업을 선택하세요 (기본값: 1번 전체 테스트):"
echo "   [1] 기본 비전 & YOLO 인터페이스 통합 검증 테스트"
echo "   [2] [이재환 파트] 사과 품질 YOLO 모델 학습 시작 (train_apple_yolo.py)"
echo "   [3] [이재환 파트] 실시간 사과 품질 추론 테스트 (test_yolo_infer.py)"
echo "   [4] bash 쉘 유지 (가상환경 활성화 상태로 터미널 진입)"
echo "======================================================================"
read -t 10 -p "선택 (1/2/3/4, 10초 후 1번 자동 실행): " choice
choice=${choice:-1}

case $choice in
    1)
        echo -e "\n▶ [1번] 통합 파이프라인 검증 시작..."
        python3 test_quality_pipeline.py
        echo -e "\n▶ [YOLO 파트] 실시간 추론 인터페이스 검증..."
        cd yolo_quality && python3 test_yolo_infer.py
        echo -e "\n🎉 모든 검증이 성공적으로 완료되었습니다!"
        ;;
    2)
        echo -e "\n▶ [2번] YOLO 사과 품질 모델 학습 시작 (50 Epochs)..."
        cd yolo_quality
        python3 train_apple_yolo.py --data apple_data.yaml --model yolo11n.pt --epochs 50 --batch 16 --device 0
        echo -e "\n🎉 학습 완료! 결과 그래프 및 가중치 확인이 가능합니다."
        read -p "엔터를 누르면 메뉴로 돌아갑니다..."
        ;;
    3)
        echo -e "\n▶ [3번] YOLO 추론 인터페이스 단독 실행..."
        cd yolo_quality
        python3 test_yolo_infer.py
        ;;
    4)
        echo -e "\n▶ [4번] 가상환경 터미널을 유지합니다. 자유롭게 명령어를 입력하세요."
        exec bash --norc -i
        ;;
    *)
        echo "잘못된 입력입니다. 1번을 기본 실행합니다."
        python3 test_quality_pipeline.py
        cd yolo_quality && python3 test_yolo_infer.py
        ;;
esac
