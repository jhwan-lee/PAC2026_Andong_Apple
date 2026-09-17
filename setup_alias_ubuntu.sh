#!/bin/bash
# ==============================================================================
# 우분투 어디서든 'pac' 한 단어만 입력하면 실행되도록 ~/.bashrc 에 alias 등록
# 사용법: bash setup_alias_ubuntu.sh
# ==============================================================================

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
RUN_SCRIPT="$PROJECT_DIR/run_pac.sh"

# 실행 권한 부여
chmod +x "$RUN_SCRIPT"

# ~/.bashrc 에 alias 등록 여부 확인
if grep -q "alias pac=" ~/.bashrc; then
    # 기존 등록된 경로 갱신
    sed -i "/alias pac=/d" ~/.bashrc
fi

echo "alias pac='bash \"$RUN_SCRIPT\"'" >> ~/.bashrc
echo "======================================================================"
echo "🎉 등록 완료! 이제 우분투 터미널 어디서든 아래 한 단어만 치면 실행됩니다:"
echo ""
echo "    pac"
echo ""
echo "현재 터미널에 즉시 적용하려면: source ~/.bashrc"
echo "======================================================================"
