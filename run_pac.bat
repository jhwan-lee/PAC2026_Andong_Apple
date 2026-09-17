@echo off
chcp 65001 > nul
setlocal

echo ======================================================================
echo 🍎 PAC 2026 안동청과 미션 - 윈도우 자동 실행 시스템
echo ======================================================================

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

if not exist ".venv" (
    echo [1/4] 파이썬 가상환경 생성 중...
    python -m venv .venv
)

call .venv\Scripts\activate.bat
echo [2/4] 가상환경 활성화 완료!

python -c "import ultralytics, cv2, torch" 2>nul
if %errorlevel% neq 0 (
    echo [3/4] 필수 패키지 설치 진행 중...
    pip install -r requirements.txt
)

echo [4/4] 통합 검증 파이프라인 실행!
python test_quality_pipeline.py
cd yolo_quality
python test_yolo_infer.py

pause
