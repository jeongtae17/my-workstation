# -*- coding: utf-8 -*-
import json
from stock_analyzer.config import client

def analyze_ai(
    ticker,
    signal,
    score,
    rsi,
    macd,
    prob,
    news,
    financials=None
):
    headlines = "\n".join(
        [f"- {n.title}" for n in news]
    )
    
    fin_text = ""
    if financials and financials.get("history"):
        fin_text = "\n[재무 데이터 (최근 3~4년)]\n"
        for h in financials["history"]:
            fin_text += f"- {h['year']}: 매출 {h['revenue']:,}, 영업이익 {h['op_income']:,}\n"
        fin_text += f"\n[어닝 요약]\n{financials.get('earnings_summary', 'N/A')}\n"

    prompt = f"""
티커: {ticker}

기술 분석:
- Signal: {signal}
- Score: {score}
- RSI: {rsi:.2f}
- MACD: {macd:.2f}
- 상승 확률: {prob}%
{fin_text}
뉴스:
{headlines}

위의 기술적 지표, 재무 데이터(매출/영업이익 성장세), 어닝 결과를 종합적으로 분석하여 투자 전략을 제시하세요.
아래 JSON 형식으로만 답변:

{{
"summary":"재무와 기술적 관점을 통합한 구체적인 분석 결과",
"risk":"LOW/MID/HIGH",
"style":"단기/스윙/장기"
}}
"""

    try:
        rsp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.3
        )

        return json.loads(
            rsp.choices[0].message.content
        )

    except:
        return {
            "summary": "AI 분석 실패",
            "risk": "MID",
            "style": "단기"
        }
