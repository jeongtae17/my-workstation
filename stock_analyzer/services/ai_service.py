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
- MACD Intensity (강도): {(macd / financials.get('current_price', 1) * 100) if financials and financials.get('current_price') else 'N/A'}%
- 상승 확률: {prob}%
{fin_text}
뉴스:
{headlines}

주의: MACD 수치가 크게 느껴질 수 있으나, 이는 주가 절대값이 높은 한국 종목(예: 5만원 이상)에서 정상적인 현상입니다. MACD 강도(Intensity) 비율을 참고하여 과매도/과매수 여부를 판단하세요.

위의 기술적 지표, 재무 데이터, 그리고 제공된 최신 뉴스 헤드라인들을 종합적으로 분석하여 투자 전략을 제시하세요.
특히 뉴스 내용에서 시장의 반응이나 주요 이슈를 파악하여 'summary'에 함께 정리해 포함시켜야 합니다.

아래 JSON 형식으로만 답변:

{{
"summary":"재무, 기술적 관점 및 최신 뉴스 분석을 통합한 구체적인 분석 결과",
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
