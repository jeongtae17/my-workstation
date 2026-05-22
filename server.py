# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn  # type: ignore
import os
import re
import json
import requests
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

# 파일 기반 티커 로드 로직
TICKER_FILE = os.path.join(os.path.dirname(__file__), "fixed_tickers.json")

def normalize_ticker_key(name: str) -> str:
    normalized = name.strip()
    normalized = re.sub(r"[\s\u3000\-_,./()\[\]{}'\"]+", "", normalized)
    normalized = normalized.upper()
    normalized = re.sub(r"(주식회사|유한회사|합자회사|회사|주식|주|코퍼레이션|CORP|INC|LTD|PLC)$", "", normalized)
    return normalized

def load_ticker_map():
    # 1. 기본 인덱스 및 주요 약칭 맵
    base_map = {
        "코스피": "^KS11", "KOSPI": "^KS11", "코스닥": "^KQ11", "KOSDAQ": "^KQ11",
        "나스닥": "^IXIC", "NASDAQ": "^IXIC", "S&P 500": "^GSPC", "S&P500": "^GSPC",
        "하닉": "000660.KS", "삼전": "005930.KS", "포홀": "005490.KS", "곱버스": "252670.KS"
    }
    # 2. 파일 로드 (전체 국내 주식 리스트)
    if os.path.exists(TICKER_FILE):
        try:
            with open(TICKER_FILE, "r", encoding="utf-8") as f:
                file_map = json.load(f)
                base_map.update(file_map)
        except Exception as e:
            print(f"Error loading ticker file: {e}")
    return base_map

KOREAN_TICKER_MAP = load_ticker_map()
NORMALIZED_TICKER_MAP = {normalize_ticker_key(k): v for k, v in KOREAN_TICKER_MAP.items()}

def get_ticker_by_keyword(normalized_name: str):
    if "HANWHA" in normalized_name or "한화" in normalized_name:
        if "ENGINE" in normalized_name or "엔진" in normalized_name: return "082740.KS"
        if any(kw in normalized_name for kw in ["손해", "보험", "INSURANCE"]): return "000370.KS"
        if any(kw in normalized_name for kw in ["에어로", "AERO"]): return "012450.KS"
    return None

def search_ticker_by_name(user_input: str):
    query = user_input.strip()
    if not query:
        return None

    # 이미 정확한 티커 심볼을 입력한 경우 우선 검증
    upper_query = query.upper()
    if validate_ticker(upper_query):
        return upper_query

    # Yahoo Finance 검색 API를 활용한 보강 검색
    try:
        url = "https://query2.finance.yahoo.com/v1/finance/search"
        params = {"q": query, "quotesCount": 20, "newsCount": 0}
        rsp = requests.get(url, params=params, timeout=10)
        data = rsp.json()
        for quote in data.get("quotes", []):
            symbol = quote.get("symbol")
            if symbol and validate_ticker(symbol):
                return symbol.upper()
    except Exception:
        pass

    # yfinance 기본 검색도 보조로 유지
    try:
        search = yf.Search(query)
        for quote in search.quotes:
            symbol = quote.get("symbol")
            if symbol and validate_ticker(symbol):
                return symbol.upper()
    except Exception:
        pass

    return None

def resolve_ticker_with_gpt(user_input: str):
    """지능형 티커 추론 (다국어 및 시장 우선순위 대응)"""
    try:
        clean_input = user_input.strip()
        rsp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an elite Yahoo Finance ticker normalization engine.\n\n"

                            "Your ONLY responsibility is converting arbitrary user input into the exact Yahoo Finance ticker symbol.\n\n"

                            "You MUST return ONLY the ticker symbol string.\n"
                            "No explanations.\n"
                            "No markdown.\n"
                            "No labels.\n"
                            "No JSON.\n"
                            "No punctuation outside the ticker.\n"
                            "No whitespace before or after output.\n\n"

                            "========================\n"
                            "[PRIMARY OBJECTIVE]\n"
                            "========================\n"

                            "Resolve the intended financial instrument as accurately as possible.\n"
                            "Prioritize real-world market conventions over literal text matching.\n\n"

                            "========================\n"
                            "[CORE RESOLUTION RULES]\n"
                            "========================\n"

                            "1. Resolve by PRIMARY HOME MARKET.\n"
                            "   - Ignore the language of the query.\n"
                            "   - Ignore the nationality of the user.\n"
                            "   - Ignore keyboard layout differences.\n"
                            "   - Determine where the company is mainly listed and traded.\n\n"

                            "2. MARKET SUFFIX RULES:\n"
                            "   - United States -> raw ticker only\n"
                            "   - Korea KOSPI -> .KS\n"
                            "   - Korea KOSDAQ -> .KQ\n"
                            "   - Japan -> .T\n"
                            "   - Hong Kong -> .HK\n"
                            "   - Shanghai -> .SS\n"
                            "   - Shenzhen -> .SZ\n"
                            "   - Taiwan -> .TW\n"
                            "   - London -> .L\n"
                            "   - Toronto -> .TO\n"
                            "   - Euronext Paris -> .PA\n"
                            "   - Frankfurt/Xetra -> .DE\n"
                            "   - Australia ASX -> .AX\n\n"

                            "3. DUAL-LISTING RULE:\n"
                            "   - Always prioritize the home-country listing.\n"
                            "   - Use ADR/US listing ONLY if explicitly requested.\n"
                            "   - If user says 'US', 'ADR', or 'NASDAQ/NYSE', prefer US ticker.\n\n"

                            "Examples:\n"
                            "   - Alibaba -> 9988.HK\n"
                            "   - Alibaba US -> BABA\n"
                            "   - Sony -> 6758.T\n"
                            "   - Sony ADR -> SONY\n"
                            "   - TSMC -> 2330.TW\n"
                            "   - TSMC NYSE -> TSM\n"
                            "   - Samsung Electronics -> 005930.KS\n\n"

                            "========================\n"
                            "[MARKET INDEX PRIORITY]\n"
                            "========================\n"

                            "Map major indices EXACTLY:\n"
                            "   - KOSPI -> ^KS11\n"
                            "   - KOSDAQ -> ^KQ11\n"
                            "   - S&P 500 -> ^GSPC\n"
                            "   - SP500 -> ^GSPC\n"
                            "   - Nasdaq -> ^IXIC\n"
                            "   - Nasdaq Composite -> ^IXIC\n"
                            "   - Nasdaq 100 -> ^NDX\n"
                            "   - Dow Jones -> ^DJI\n"
                            "   - Russell 2000 -> ^RUT\n"
                            "   - Nikkei -> ^N225\n"
                            "   - Nikkei 225 -> ^N225\n"
                            "   - Hang Seng -> ^HSI\n"
                            "   - DAX -> ^GDAXI\n"
                            "   - FTSE 100 -> ^FTSE\n"
                            "   - CAC 40 -> ^FCHI\n\n"

                            "========================\n"
                            "[ENTITY MATCHING LOGIC]\n"
                            "========================\n"

                            "1. Accept multilingual company names.\n"
                            "2. Accept abbreviations and aliases.\n"
                            "3. Accept informal retail-investor nicknames.\n"
                            "4. Accept common misspellings if intent is obvious.\n"
                            "5. Accept partial names when uniquely identifiable.\n"
                            "6. Prefer operating companies over ETFs.\n"
                            "7. Prefer listed parent company over subsidiaries unless explicitly specified.\n"
                            "8. If multiple companies share the same name, choose the globally dominant listed entity.\n\n"

                            "Examples:\n"
                            "   - Apple -> AAPL\n"
                            "   - 애플 -> AAPL\n"
                            "   - 엔비디아 -> NVDA\n"
                            "   - Nvidia -> NVDA\n"
                            "   - 아이온큐 -> IONQ\n"
                            "   - 테슬라 -> TSLA\n"
                            "   - 삼성전자 -> 005930.KS\n"
                            "   - 삼전 -> 005930.KS\n"
                            "   - 카카오 -> 035720.KS\n"
                            "   - 네이버 -> 035420.KS\n"
                            "   - 쿠팡 -> CPNG\n\n"

                            "========================\n"
                            "[SPECIAL SECURITY TYPES]\n"
                            "========================\n"

                            "Handle these correctly:\n"
                            "   - ETFs\n"
                            "   - REITs\n"
                            "   - Preferred shares\n"
                            "   - ADRs\n"
                            "   - Leveraged ETFs\n"
                            "   - Inverse ETFs\n"
                            "   - Closed-end funds\n\n"

                            "Examples:\n"
                            "   - QQQ -> QQQ\n"
                            "   - SOXL -> SOXL\n"
                            "   - TQQQ -> TQQQ\n"
                            "   - 삼성전자우 -> 005935.KS\n\n"

                            "========================\n"
                            "[AMBIGUITY RESOLUTION]\n"
                            "========================\n"

                            "If input is ambiguous:\n"
                            "1. Prefer the most liquid and globally recognized ticker.\n"
                            "2. Prefer active listings over delisted companies.\n"
                            "3. Prefer common equity over bonds or derivatives.\n"
                            "4. Prefer companies over funds when ambiguity exists.\n\n"

                            "========================\n"
                            "[INVALID ENTITY RULES]\n"
                            "========================\n"

                            "Return INVALID for:\n"
                            "   - Sports teams\n"
                            "   - Esports teams\n"
                            "   - Celebrities\n"
                            "   - Politicians\n"
                            "   - Fictional characters\n"
                            "   - Generic nouns\n"
                            "   - Countries\n"
                            "   - Religions\n"
                            "   - Non-financial concepts\n"
                            "   - Private companies without public ticker\n"
                            "   - Undefined entities\n\n"

                            "Examples:\n"
                            "   - Hanwha Eagles -> INVALID\n"
                            "   - T1 Faker -> INVALID\n"
                            "   - Elon Musk -> INVALID\n"
                            "   - Naruto -> INVALID\n"
                            "   - Bitcoin mining -> INVALID\n\n"

                            "========================\n"
                            "[STRICT OUTPUT POLICY]\n"
                            "========================\n"

                            "Return EXACTLY ONE STRING.\n"
                            "No quotes.\n"
                            "No code block.\n"
                            "No newline.\n"
                            "No commentary.\n\n"

                            "VALID OUTPUTS:\n"
                            "AAPL\n"
                            "005930.KS\n"
                            "^GSPC\n"
                            "9988.HK\n"
                            "TSM\n"
                            "INVALID\n\n"

                            "INVALID OUTPUTS:\n"
                            "Ticker: AAPL\n"
                            "`AAPL`\n"
                            "{\"ticker\":\"AAPL\"}\n"
                            "The ticker is AAPL\n"
                            "AAPL stock\n\n"

                            "========================\n"
                            "[FINAL SAFETY RULE]\n"
                            "========================\n"

                            "If confidence is low or entity cannot be mapped reliably, return INVALID.\n\n"

                            "1. Prefer ACTIVE listings over delisted securities.\n"
                            "2. Ignore historical or inactive tickers unless explicitly requested.\n"
                            "3. If both active and delisted entities share the same name:\n"
                            "   - prioritize the currently traded company.\n"
                            "4. Never return obsolete Yahoo Finance symbols if an active listing exists."
                        )
                    },
                {"role": "user", "content": f"Ticker for: {clean_input}"}
            ],
            temperature=0,
        )
        ticker = rsp.choices[0].message.content.strip().upper()
        ticker = "".join(c for c in ticker if c.isalnum() or c in ".-").strip()
        return ticker if (ticker and len(ticker) <= 12) else None
    except: return None

def resolve_ticker(user_input: str):
    """순서: [1] 파일/하드코딩 맵 -> [2] 키워드 매칭 -> [3] GPT 추론 -> [4] 최종 검색"""
    clean_input = user_input.strip()
    upper_input = clean_input.upper()
    normalized_input = normalize_ticker_key(clean_input)

    # 1. 로컬 데이터베이스(fixed_tickers.json 및 기본 맵) 검색
    if clean_input in KOREAN_TICKER_MAP: return KOREAN_TICKER_MAP[clean_input]
    if upper_input in KOREAN_TICKER_MAP: return KOREAN_TICKER_MAP[upper_input]
    if normalized_input in NORMALIZED_TICKER_MAP: return NORMALIZED_TICKER_MAP[normalized_input]

    # 2. 계열사 키워드 매칭
    sub_ticker = get_ticker_by_keyword(normalized_input)
    if sub_ticker: return sub_ticker

    # 3. GPT 추론 (리스트에 없는 마이너/해외 종목 대응)
    gpt_ticker = resolve_ticker_with_gpt(clean_input)
    if gpt_ticker:
        if gpt_ticker == "INVALID": return "INVALID"
        if validate_ticker(gpt_ticker): return gpt_ticker

    # 4. 최종 수단 검색
    searched = search_ticker_by_name(clean_input)
    return searched if searched else "INVALID"

def check_delisted_with_gpt(ticker_query: str, ticker: str):
    try:
        rsp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "종목 상태 확인 전문가입니다. 상장 유지(ACTIVE)인지 폐지(DELISTED)인지 판별하여 한 단어로만 답하세요."},
                {"role": "user", "content": f"상태 확인: {ticker_query} ({ticker})"}
            ],
            temperature=0,
        )
        return "DELISTED" in rsp.choices[0].message.content.upper()
    except: return False

@app.get("/", response_class=HTMLResponse)
async def read_root():
    template_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    with open(template_path, "r", encoding="utf-8") as f: return f.read()

@app.get("/analyze")
async def analyze(ticker_query: str):
    try:
        # 1. 티커 식별
        ticker = resolve_ticker(ticker_query)
        if ticker == "INVALID" or not ticker:
            return {"status": "ERROR", "error": f"'{ticker_query}'은(는) 유효하지 않거나 찾을 수 없는 종목입니다."}

        # 2. 실제 주가 데이터가 있는지 확인한 뒤 분석 진행
        df = fetch_stock_data(ticker)
        if df.empty:
            return {"status": "ERROR", "error": "주가 데이터를 불러오는 데 실패했습니다."}

        company = fetch_company_info(ticker)
        df = calculate_indicators(df)
        signal, score = generate_signal(df)
        rsi = df["RSI"].iloc[-1]
        macd = df["MACD"].iloc[-1]
        macd_intensity = (macd / df["Close"].iloc[-1]) * 100
        prob = calculate_probability(score, rsi, macd, df)
        news = fetch_news(ticker, company.get("name", ""))
        financials = fetch_financials(ticker)
        financials['current_price'] = df["Close"].iloc[-1]

        ai_result = analyze_ai(ticker, signal, score, rsi, macd, prob, news, financials)
        rule_style = determine_style(company, df, rsi)

        return {
            "status": "SUCCESS",
            "ticker": ticker,
            "company": company,
            "signal": signal,
            "score": score,
            "rsi": float(rsi),
            "macd": float(macd),
            "macd_intensity": float(macd_intensity),
            "probability": prob,
            "financials": financials,
            "rule_style": rule_style,
            "news": [{"title": n.title, "url": n.url} for n in news],
            "ai_analysis": ai_result
        }
    except Exception as e:
        print(f"Server Error: {e}")
        return {"status": "ERROR", "error": f"서버 분석 중 예외 발생: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
