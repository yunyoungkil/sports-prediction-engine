"""
telegram_bot.py
===============
신규 회차 분석이 완료되면 최고 기대가치(EV) TOP 3 추천 픽을 
스마트폰 텔레그램 메신저로 자동 발송하는 무료 푸시 알림 모듈입니다.
"""

import os
import sys
import json
import urllib.request
import urllib.parse
from pathlib import Path

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def send_telegram_alert(
    predictions_path: str = "data/predictions.json",
    bot_token: str = None,
    chat_id: str = None
) -> bool:
    """
    최신 예측 결과에서 가장 기대가치가 높은 TOP 3 경기를 텔레그램으로 전송합니다.
    환경변수 TELEGRAM_BOT_TOKEN 및 TELEGRAM_CHAT_ID를 우선 사용합니다.
    """
    token = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN")
    cid = chat_id or os.environ.get("TELEGRAM_CHAT_ID")
    
    if not token or not cid:
        print("\nℹ [텔레그램 알림 안내]")
        print("  텔레그램 봇 토큰 또는 Chat ID가 설정되지 않았습니다.")
        print("  • 텔레그램 앱에서 @BotFather 검색 후 /newbot 입력 (1분 소요)")
        print("  • 환경변수 TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID를 등록하시면 스마트폰 무료 알림이 켜집니다.")
        return False

    pred_file = Path(predictions_path)
    if not pred_file.exists():
        print(f"⚠ 예측 데이터 파일 없음: {pred_file}")
        return False

    with open(pred_file, "r", encoding="utf-8") as f:
        matches = json.load(f)

    # 추천 경기(is_recommended) 중 기대가치 상위 3개 선정
    recommended = [m for m in matches if m.get("evaluation", {}).get("is_recommended")]
    recommended.sort(key=lambda x: x["evaluation"]["best_ev"], reverse=True)
    top_picks = recommended[:3]

    if not top_picks:
        text = "⚽ [스포츠토토 AI]\n이번 회차는 기대가치(EV > 0)가 기준을 넘는 경기가 없습니다. (패스 권장)"
    else:
        text = "🔥 <b>[스포츠토토 AI] 이번 회차 최고 가치 픽 TOP 3</b>\n\n"
        for i, m in enumerate(top_picks, 1):
            eval_info = m["evaluation"]
            odds = m["betman_odds"][eval_info["best_selection"]]
            prob = m["model_probabilities"][eval_info["best_selection"]] * 100
            
            text += f"<b>{i}️⃣ [{m['league']}] {m['home_team']} vs {m['away_team']}</b>\n"
            text += f"👉 추천: <b>{eval_info['best_selection_name']}</b> ({odds}배)\n"
            text += f"📊 AI 승률: {prob:.1f}% | 기대가치: <b>{eval_info['best_ev_pct']}</b>\n"
            text += f"⏰ 경기일시: {m['match_time']}\n\n"
        
        text += "💡 <i>Dixon-Coles 수치 통계 모델 기반 자동 분석</i>"

    # Telegram SendMessage API 호출
    api_url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": cid,
        "text": text,
        "parse_mode": "HTML"
    }
    
    try:
        data = urllib.parse.urlencode(payload).encode("utf-8")
        req = urllib.request.Request(api_url, data=data)
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status == 200:
                print("✓ 텔레그램 스마트폰 알림 발송 완료!")
                return True
    except Exception as e:
        print(f"⚠ 텔레그램 알림 발송 실패: {e}")
        return False


def send_telegram_warning(warning_text: str, bot_token: str = None, chat_id: str = None) -> bool:
    """
    파이프라인 실행 중 오류나 캐시 폴백 발생 시 텔레그램으로 경고를 발송합니다.
    """
    token = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN")
    cid = chat_id or os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not cid:
        return False

    text = f"⚠️ <b>[스포츠토토 AI 시스템 경고]</b>\n\n{warning_text}"
    api_url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": cid,
        "text": text,
        "parse_mode": "HTML"
    }
    try:
        data = urllib.parse.urlencode(payload).encode("utf-8")
        req = urllib.request.Request(api_url, data=data)
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status == 200:
                print("✓ 텔레그램 경고 알림 발송 완료!")
                return True
    except Exception as e:
        print(f"⚠ 텔레그램 경고 발송 실패: {e}")
        return False


if __name__ == "__main__":
    send_telegram_alert()
