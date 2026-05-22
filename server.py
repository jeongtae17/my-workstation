# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
import re
import json
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
TICKER_FILE = os.path.join(os.path.dirname(__file__), "korean_tickers.json")

def normalize_ticker_key(name: str) -> str:
    normalized = name.strip()
    normalized = re.sub(r"[\s\u3000\-_,./()\[\]{}'\"]+", "", normalized)
    normalized = normalized.upper()
    normalized = re.sub(r"(주식회사|유한회사|합자회사|회사|주식|주|코퍼레이션|CORP|INC|LTD|PLC)$", "", normalized)
    return normalized

def load_ticker_map():
    # 1. 기본 인덱스 및 주요 약칭 맵 (하드코딩)
    base_map = {
        "코스피": "^KS11", "KOSPI": "^KS11", "코스닥": "^KQ11", "KOSDAQ": "^KQ11",
        "나스닥": "^IXIC", "NASDAQ": "^IXIC", "S&P 500": "^GSPC", "S&P500": "^GSPC",
        "하닉": "000660.KS", "삼전": "005930.KS", "포홀": "005490.KS", "곱버스": "252670.KS"
    }

    # 2. 파일 로드 (korean_tickers.json)
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
    """특정 키워드가 포함된 경우 해당 계열사로 매핑"""
    if "HANWHA" in normalized_name or "한화" in normalized_name:
        if "ENGINE" in normalized_name or "엔진" in normalized_name: return "082740.KS"
        if any(kw in normalized_name for kw in ["손해", "보험", "INSURANCE"]): return "000370.KS"
        if any(kw in normalized_name for kw in ["에어로", "AERO"]): return "012450.KS"
    return None

def search_ticker_by_name(user_input: str):
    try:
        search = yf.Search(user_input)
        for quote in search.quotes:
            symbol = quote.get("symbol")
            if symbol and validate_ticker(symbol): return symbol.upper()
    except: return None
    return None

def resolve_ticker_with_gpt(user_input: str):
    """GPT를 통한 지능형 티커 추론 (다국어 및 시장 우선순위 대응)"""
    try:
        clean_input = user_input.strip()
        rsp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an elite financial data engineer specializing in the Yahoo Finance ticker indexing system.\n"
                        "Your absolute priority is to convert the user's input (which may be in Korean, English, Chinese, Japanese, or any abbreviation) "
                        "into the exact Yahoo Finance ticker symbol string.\n\n"

                        "⚠️ [CRITICAL SEARCH RULES]\n"
                        "0. MARKET INDICES PRIORITY:\n"
                        "   - If the user enters a major market index (e.g., 'KOSPI', 'KOSDAQ', 'S&P 500', 'Nasdaq', 'Dow Jones'), map it to its specific ticker.\n"
                        "   - KOSPI -> '^KS11', KOSDAQ -> '^KQ11', S&P 500 -> '^GSPC', Nasdaq Composite -> '^IXIC'.\n\n"
                        "1. INTENT OVER LANGUAGE (CORE RULE):\n"
                        "   - Do NOT determine the target stock exchange based on the input language. Determine it based on the 'Primary Home Market' where the company is mainly listed and traded.\n"
                        "   - Example: If a user enters a Korean company in Japanese ('サムスン電子') or Chinese ('三星电子'), you MUST recognize it as a South Korean asset and route it to the KRX (.KS/.KQ).\n\n"
                        "2. MARKET ROUTING SUMMARY:\n"
                        "   - South Korea: Append '.KS' for KOSPI or '.KQ' for KOSDAQ. (Search KRX first if abbreviations like 'ISC' are ambiguous).\n"
                        "   - United States: Output the raw ticker WITHOUT any suffix (e.g., 'NVDA', 'TSLA', 'AAPL', 'IONQ').\n"
                        "   - Hong Kong: Append '.HK' (e.g., '0700.HK', '9988.HK').\n"
                        "   - Japan (Tokyo): Append '.T' (e.g., '7203.T').\n"
                        "   - China (Shanghai / Shenzhen): Append '.SS' or '.SZ' respectively.\n\n"
                        "3. DUAL-LISTING & CONFLICT RESOLUTION:\n"
                        "   - If a company is dual-listed in multiple countries (e.g., Alibaba is listed in both US as 'BABA' and HK as '9988.HK'), prioritize the Home Country / Asian primary liquidity exchange (Hong Kong for Chinese firms) unless the US market is explicitly requested.\n"
                        "   - Special Case: Companies like Coupang ('CPNG') whose primary listing is solely in the US, output the US raw ticker.\n\n"
                        "4. RESPONSE FORMAT:\n"
                        "   - Output ONLY the raw ticker symbol. Absolutely NO explanations, NO quotes, NO markdown, NO spaces, NO trailing periods.\n\n"
                        "5. GROUP AFFILIATE RULE:\n"
                        "   - If the user input refers to a subsidiary, affiliate, or group brand name, map it to the specific listed entity where possible rather than the parent holding company.\n"
                        "   - Example: 'Hanwha Engine' should resolve to '082740.KS' even though it is a Hanwha group affiliate.\n\n"
                        "📋 [EXACT MULTI-LANGUAGE MAPPING EXAMPLES]\n"
                        "- '더존비즈온' -> '012510.KS'\n"
                        "- 'ISC' -> '095340.KQ'\n"
                        "- '아이에스씨' -> '095340.KQ'\n"
                        "- '아이온큐' -> 'IONQ'\n"
                        "- '엔비디아' -> 'NVDA'\n"
                        "- '삼성전자' -> '005930.KS'\n"
                        "- '테슬라' -> 'TSLA'\n"
                        "- '三星电子' (Korean Co. in Chinese) -> '005930.KS'\n"
                        "- 'サムスン電子' (Korean Co. in Japanese) -> '005930.KS'\n"
                        "- 'カカオ' (Korean Co. in Japanese) -> '035720.KS'\n"
                        "- '腾讯' -> '0700.HK'\n"
                        "- 'Tencent' -> '0700.HK'\n"
                        "- '阿里巴巴' -> '9988.HK'\n"
                        "- 'Alibaba' -> '9988.HK'\n"
                        "- 'Toyota' -> '7203.T'\n"
                        "- 'トヨタ' -> '7203.T'\n"
                        "- '한화엔진' -> '082740.KS'\n"
                        "- '한화손해보험' -> '000370.KS'\n"
                        "- '엔비디아' -> 'NVDA'\n"
                        "- '아이온큐' -> 'IONQ'\n"
                        "- 'Coupang' -> 'CPNG'"
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
    user_input = user_input.strip()
    normalized_input = normalize_ticker_key(user_input)

    # 1. 메모리 맵 확인
    if user_input in KOREAN_TICKER_MAP: return KOREAN_TICKER_MAP[user_input]
    if user_input.upper() in KOREAN_TICKER_MAP: return KOREAN_TICKER_MAP[user_input.upper()]
    if normalized_input in NORMALIZED_TICKER_MAP: return NORMALIZED_TICKER_MAP[normalized_input]

    # 2. 키워드 매칭
    sub_ticker = get_ticker_by_keyword(normalized_input)
    if sub_ticker: return sub_ticker

    # 3. GPT 추론
    gpt_ticker = resolve_ticker_with_gpt(user_input)
    if gpt_ticker:
        if gpt_ticker == "INVALID": return "INVALID"
        if validate_ticker(gpt_ticker): return gpt_ticker

    # 4. 최종 검색
    searched = search_ticker_by_name(user_input)
    return searched if searched else "INVALID"

def check_delisted_with_gpt(ticker_query: str, ticker: str):
    """상장 폐지 여부 실시간 확인"""
    try:
        rsp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "종목 상태 확인 전문가입니다. 해당 티커가 상장 유지(ACTIVE)인지 폐지(DELISTED)인지 판별하세요. 한 단어로만 답하세요."},
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
        ticker = resolve_ticker(ticker_query)
        if ticker == "INVALID" or not ticker:
            return {"error": f"'{ticker_query}'은(는) 유효하지 않거나 찾을 수 없는 종목입니다.", "status": "INVALID"}

        if check_delisted_with_gpt(ticker_query, ticker):
             return {"error": f"'{ticker_query}'({ticker})은 상장 폐지된 종목입니다.", "status": "DELISTED"}

        df = fetch_stock_data(ticker)
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
            "ticker": ticker, "company": company, "signal": signal, "score": score,
            "rsi": float(rsi), "macd": float(macd), "macd_intensity": float(macd_intensity),
            "probability": prob, "financials": financials, "rule_style": rule_style,
            "news": [{"title": n.title, "url": n.url} for n in news],
            "ai_analysis": ai_result, "status": "SUCCESS"
        }
    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e), "status": "ERROR"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
