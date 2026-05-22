# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
import yfinance as yf
from dotenv import load_dotenv
from openai import OpenAI

# Import core services
from stock_analyzer.services.data_service import fetch_stock_data, fetch_company_info, fetch_news, fetch_financials, validate_ticker
from stock_analyzer.services.analysis_service import calculate_indicators, generate_signal, calculate_probability, determine_style
from stock_analyzer.services.ai_service import analyze_ai

load_dotenv()

app = FastAPI(title="Stock Analyzer API")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 종목명 -> 티커 매핑 (하드코딩 추천 리스트)
KOREAN_TICKER_MAP = {
    "삼성전자": "005930.KS", "삼전": "005930.KS", "삼성전자우": "005935.KS",
    "SK하이닉스": "000660.KS", "하이닉스": "000660.KS",
    "더존비즈온": "012515.KS", "더존": "012515.KS",
    "카카오": "035720.KS", "네이버": "035420.KS", "NAVER": "035420.KS",
    "현대차": "005380.KS", "기아": "000270.KS",
    "에코프로": "086520.KQ", "에코프로비엠": "247540.KQ",
    "포스코홀딩스": "005490.KS", "POSCO홀딩스": "005490.KS",
    "엔비디아": "NVDA", "애플": "AAPL", "테슬라": "TSLA", "마이크로소프트": "MSFT",
    "구글": "GOOGL", "아마존": "AMZN", "메타": "META", "아이온큐": "IONQ"
}

def resolve_ticker_with_gpt(user_input: str):
    """정교화된 시스템 컨텍스트 규칙에 맞춰 한글/영문명을 야후 파이낸스 규격 티커로 매칭합니다."""
    try:
        clean_input = user_input.strip()

        rsp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an elite financial data engineer specializing in the Yahoo Finance ticker indexing system.\n"
                        "Your absolute priority is to convert the user's input (Korean company name, English name, or abbreviation) "
                        "into the exact Yahoo Finance ticker symbol string.\n\n"
                        
                        "⚠️ [CRITICAL SEARCH RULES]\n"
                        "1. KOREAN LISTED COMPANY:\n"
                        "   - If the input represents a company primarily active or listed in Korea (e.g., '더존', '아이에스씨', 'ISC', '한화'), "
                        "     you MUST find its 6-digit stock code and attach '.KS' (KOSPI) or '.KQ' (KOSDAQ).\n"
                        "   - IMPORTANT: Abbreviations like 'ISC' are very often Korean companies (095340.KQ). Search KRX market first for these.\n"
                        "2. US & INTERNATIONAL LISTED COMPANY:\n"
                        "   - If the company is clearly a global giant (NVDA, TSLA, AAPL, MSFT), provide the raw ticker.\n"
                        "3. RESPONSE FORMAT:\n"
                        "   - Output ONLY the raw ticker symbol. Absolutely NO explanations, NO quotes, NO markdown, NO spaces.\n\n"

                        "📋 [EXACT MAPPING EXAMPLES]\n"
                        "- '더존비즈온' -> '012515.KS'\n"
                        "- 'ISC' -> '095340.KQ'\n"
                        "- '아이에스씨' -> '095340.KQ'\n"
                        "- '아이온큐' -> 'IONQ'\n"
                        "- '엔비디아' -> 'NVDA'\n"
                        "- '삼성전자' -> '005930.KS'\n"
                        "- '테슬라' -> 'TSLA'"
                    )
                },
                {"role": "user", "content": f"Ticker for: {clean_input}"}
            ],
            temperature=0,
        )

        ticker = rsp.choices[0].message.content.strip().upper()
        # 불필요한 따옴표나 공백 제거
        ticker = "".join(c for c in ticker if c.isalnum() or c in ".-")

        if not ticker or len(ticker) > 12:
            return None

        return ticker
    except:
        return None

def resolve_ticker(user_input: str):
    user_input = user_input.strip()

    # 1. 하드코딩 맵 우선 확인
    if user_input in KOREAN_TICKER_MAP:
        return KOREAN_TICKER_MAP[user_input]
    if user_input.upper() in KOREAN_TICKER_MAP:
        return KOREAN_TICKER_MAP[user_input.upper()]

    # 2. 영문 1-5자면 '미국 주식'일 가능성을 먼저 체크하되, GPT의 판단을 보조적으로 활용함
    # "ISC" 같은 경우 미국에도 티커가 있을 수 있으나, 한국 사용자는 보통 한국 종목을 원함
    
    # 3. GPT 검색 + 유효성 검증
    gpt_ticker = resolve_ticker_with_gpt(user_input)
    if gpt_ticker and validate_ticker(gpt_ticker):
        return gpt_ticker
    
    # 4. GPT가 실패하거나 유효하지 않은 경우 마지막 수단으로 입력값 그대로 시도
    final_attempt = user_input.upper()
    return final_attempt

@app.get("/", response_class=HTMLResponse)
async def read_root():
    template_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()

@app.get("/analyze")
async def analyze(ticker_query: str):
    try:
        ticker = resolve_ticker(ticker_query)
        df = fetch_stock_data(ticker)
        company = fetch_company_info(ticker)
        df = calculate_indicators(df)
        signal, score = generate_signal(df)
        rsi = df["RSI"].iloc[-1]
        macd = df["MACD"].iloc[-1]

        # MACD 강도 비율 계산 (MACD / 현재가 * 100)
        macd_intensity = (macd / df["Close"].iloc[-1]) * 100

        prob = calculate_probability(score, rsi, macd, df)
        news = fetch_news(ticker, company.get("name", ""))
        financials = fetch_financials(ticker)

        # 주가 정보 추가 (MACD 분석용)
        financials['current_price'] = df["Close"].iloc[-1]

        ai_result = analyze_ai(ticker, signal, score, rsi, macd, prob, news, financials)
        rule_style = determine_style(company, df, rsi)

        return {
            "ticker": ticker,
            "company": company,
            "signal": signal,
            "score": score,
            "rsi": float(rsi),
            "macd": float(macd),
            "macd_intensity": float(macd_intensity),
            "probability": prob,
            "financials": financials,
            "news": [{"title": n.title, "url": n.url} for n in news],
            "ai_analysis": ai_result,
            "rule_style": rule_style
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)