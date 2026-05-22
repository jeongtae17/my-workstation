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
from stock_analyzer.services.data_service import fetch_stock_data, fetch_company_info, fetch_news, fetch_financials
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

def resolve_ticker_with_gpt(user_input: str):
    try:
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
                        "1. KOREAN COMPANY PRIORITY:\n"
                        "   - If the input is written in Korean (e.g., '더존비즈온', '삼성전자') OR represents a Korean listed company via English translation/romanization (e.g., 'Douzone', 'NAVER', 'ISC'), "
                        "     you MUST find its 6-digit stock code listed on the Korea Exchange (KRX) and attach '.KS' (KOSPI) or '.KQ' (KOSDAQ).\n"
                        "   - NEVER map a clear Korean company name to a US or international ticker just because the spelling or sound is similar.\n\n"
                        "2. RESPONSE FORMAT:\n"
                        "   - Output ONLY the raw ticker symbol (e.g., '012515.KS', 'NVDA', '005930.KS').\n"
                        "   - Absolutely NO explanations, NO quotes, NO markdown, NO spaces, NO trailing periods.\n\n"
                        "3. TEMPERATURE CONSTRAINT:\n"
                        "   - You must act deterministically. Do not guess non-existent tickers. If you are certain it is a Korean stock but the exact suffix is ambiguous, default to the most accurate historical exchange registration.\n\n"
                        
                        "🧠 [THINKING PROCESS & EXEMPLARS]\n"
                        "- User: '더존비즈온' -> Romanized: Douzone Bizon -> Korean KOSPI listed software company -> Ticker: '012515.KS'\n"
                        "- User: 'ISC' -> Korean semiconductor test socket manufacturer listed on KOSDAQ -> Ticker: '095340.KQ'\n"
                        "- User: '한화에어로스페이스' -> Hanwha Aerospace -> KOSPI listed defense company -> Ticker: '012450.KS'\n"
                        "- User: '카카오' -> Kakao -> KOSPI listed -> Ticker: '035720.KS'\n"
                        "- User: 'Nvidia' -> US listed global tech company -> Ticker: 'NVDA'\n\n"
                        
                        "📋 [EXACT MAPPING EXAMPLES]\n"
                        "- '더존비즈온' -> '012515.KS'\n"
                        "- '더존' -> '012515.KS'\n"
                        "- 'ISC' -> '095340.KQ'\n"
                        "- '삼성전자' -> '005930.KS'\n"
                        "- '에코프로비엠' -> '247540.KQ'\n"
                        "- 'Apple' -> 'AAPL'\n"
                        "- 'TSLA' -> 'TSLA'"
                    )
                },
                {"role": "user", "content": f"다음의 Yahoo Finance 티커를 알려줘: {user_input}"}
            ],
            temperature=0,
        )
        ticker = rsp.choices[0].message.content.strip().upper()
        return ticker if " " not in ticker else None
    except:
        return None

def resolve_ticker(user_input: str):
    user_input = user_input.strip()
    if 1 <= len(user_input) <= 5 and user_input.isalpha() and user_input.isascii():
        return user_input.upper()
    
    gpt_ticker = resolve_ticker_with_gpt(user_input)
    if gpt_ticker:
        return gpt_ticker
    
    return user_input.upper()

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