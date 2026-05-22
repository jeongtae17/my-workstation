# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
import re
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

# 종목명 -> 티커 매핑 (가장 정확한 1:1 매핑 리스트)
KOREAN_TICKER_MAP = {
    "코스피": "^KS11", "KOSPI": "^KS11",
    "코스닥": "^KQ11", "KOSDAQ": "^KQ11",
    "나스닥": "^IXIC", "NASDAQ": "^IXIC", "나스닥종합": "^IXIC",
    "S&P 500": "^GSPC", "S&P500": "^GSPC", "에스앤피500": "^GSPC",
    "다우": "^DJI", "다우존스": "^DJI",
    "KODEX 200": "069500.KS", "KODEX200": "069500.KS",
    "TIGER 200": "102110.KS", "TIGER200": "102110.KS",
    "KODEX 레버리지": "122630.KS", "KODEX200레버리지": "122630.KS",
    "KODEX 200선물인버스2X": "252670.KS", "곱버스": "252670.KS",
    "삼성전자": "005930.KS", "삼전": "005930.KS", "삼성전자우": "005935.KS",
    "SK하이닉스": "000660.KS", "하이닉스": "000660.KS", "에스케이하이닉스": "000660.KS", "하닉": "000660.KS", "에스케이하닉": "000660.KS",
    "더존비즈온": "012510.KS", "더존": "012510.KS",
    "아이에스씨": "095340.KQ", "ISC": "095340.KQ",
    "카카오": "035720.KS", "네이버": "035420.KS", "NAVER": "035420.KS",
    "현대차": "005380.KS", "기아": "000270.KS",
    "한화": "000880.KS", "HANWHA": "000880.KS",
    "에코프로": "086520.KQ", "에코프로비엠": "247540.KQ",
    "포스코홀딩스": "005490.KS", "POSCO홀딩스": "005490.KS", "포홀": "005490.KS",
    "엔비디아": "NVDA", "애플": "AAPL", "테슬라": "TSLA", "마이크로소프트": "MSFT",
    "구글": "GOOGL", "아마존": "AMZN", "메타": "META", "아이온큐": "IONQ",
    "한화손해보험": "000370.KS", "한화손보": "000370.KS", "HANWHA GENERAL INSURANCE": "000370.KS",
    "한화엔진": "082740.KS", "한화 엔진": "082740.KS", "HANWHA ENGINE": "082740.KS"
}

def normalize_ticker_key(name: str) -> str:
    normalized = name.strip()
    normalized = re.sub(r"[\s\u3000\-_,./()\[\]{}'\"]+", "", normalized)
    normalized = normalized.upper()
    normalized = re.sub(r"(주식회사|유한회사|합자회사|회사|주식|주|코퍼레이션|CORP|INC|LTD|PLC)$", "", normalized)
    return normalized

NORMALIZED_TICKER_MAP = {normalize_ticker_key(k): v for k, v in KOREAN_TICKER_MAP.items()}

def get_ticker_by_keyword(normalized_name: str):
    """
    특정 키워드가 명시적으로 포함된 경우에만 해당 계열사로 매핑합니다.
    """
    if "HANWHA" in normalized_name or "한화" in normalized_name:
        if "ENGINE" in normalized_name or "엔진" in normalized_name:
            return "082740.KS"
        if any(kw in normalized_name for kw in ["손해", "보험", "INSURANCE", "GENERAL"]):
            return "000370.KS"
        if any(kw in normalized_name for kw in ["에어로", "AERO", "SPACE"]):
            return "012450.KS"
        if any(kw in normalized_name for kw in ["솔루션", "SOLUTION"]):
            return "009830.KS"
        if any(kw in normalized_name for kw in ["오션", "OCEAN"]):
            return "042660.KS"
    return None

def search_ticker_by_name(user_input: str):
    try:
        search = yf.Search(user_input)
        for quote in search.quotes:
            symbol = quote.get("symbol")
            if symbol and validate_ticker(symbol):
                return symbol.upper()
    except:
        return None
    return None

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
                        "Your absolute priority is to convert the user's input into the exact Yahoo Finance ticker symbol string.\n\n"
                        
                        "⚠️ [CRITICAL SEARCH RULES]\n"
                        "1. SEARCH STRATEGY (Investing.com Style):\n"
                        "   - For any Korean stock, your absolute priority is to identify its official 6-digit KRX numeric code (as used on Investing.com or KRX).\n"
                        "   - Once identified, append '.KS' for KOSPI or '.KQ' for KOSDAQ to match Yahoo Finance format.\n"
                        "2. MARKET INDICES:\n"
                        "   - KOSPI -> '^KS11', KOSDAQ -> '^KQ11', S&P 500 -> '^GSPC', Nasdaq -> '^IXIC'.\n"
                        "3. KOREAN LISTED COMPANY:\n"
                        "   - Priority for minor/small-cap/bio stocks: Find their 6-digit KRX code (e.g., '인투셀' -> 214610.KQ).\n"
                        "4. GROUP AFFILIATE VS HOLDING COMPANY:\n"
                        "   - If input is generic (e.g., 'Hanwha', 'Samsung'), prioritize the Parent/Main entity.\n"
                        "5. INVALID & NON-STOCK ENTITIES (STRICT):\n"
                        "   - If input is a sports team (e.g., 'Eagles', 'Hanwha Eagles', 'Damwon', 'T1'), a person, or non-stock, output 'INVALID'.\n"
                        "6. RESPONSE FORMAT:\n"
                        "   - Output ONLY the raw ticker symbol. Absolutely NO explanations, NO quotes, NO markdown, NO spaces.\n\n"

                        "📋 [EXACT MAPPING EXAMPLES]\n"
                        "- '더존비즈온' -> '012510.KS'\n"
                        "- '인투셀' -> '214610.KQ'\n"
                        "- '아이에스씨' / 'ISC' -> '095340.KQ'\n"
                        "- '한화' -> '000880.KS'\n"
                        "- '한화이글스' -> 'INVALID'\n"
                        "- '담원' -> 'INVALID'\n"
                        "- '아이온큐' -> 'IONQ'"
                    )
                },
                {"role": "user", "content": f"Ticker for: {clean_input}"}
            ],
            temperature=0,
        )

        ticker = rsp.choices[0].message.content.strip().upper()
        ticker = "".join(c for c in ticker if c.isalnum() or c in ".-").strip()

        if not ticker or len(ticker) > 12:
            return None

        return ticker
    except:
        return None

def resolve_ticker(user_input: str):
    user_input = user_input.strip()
    normalized_input = normalize_ticker_key(user_input)

    # 1. 하드코딩 맵 우선 확인 (KODEX 200, 하닉, 나스닥 등)
    if user_input in KOREAN_TICKER_MAP:
        return KOREAN_TICKER_MAP[user_input]
    if user_input.upper() in KOREAN_TICKER_MAP:
        return KOREAN_TICKER_MAP[user_input.upper()]
    if normalized_input in NORMALIZED_TICKER_MAP:
        return NORMALIZED_TICKER_MAP[normalized_input]

    # 2. 보조 키워드 매칭 (자회사 대응)
    subsidiary_ticker = get_ticker_by_keyword(normalized_input)
    if subsidiary_ticker:
        return subsidiary_ticker

    # 3. GPT 검색 + 유효성 검증
    gpt_ticker = resolve_ticker_with_gpt(user_input)
    if gpt_ticker:
        if gpt_ticker == "INVALID":
            return "INVALID"
        if validate_ticker(gpt_ticker):
            return gpt_ticker

    # 4. yfinance 검색
    searched_ticker = search_ticker_by_name(user_input)
    if searched_ticker:
        return searched_ticker

    # 5. 최종 실패 시 INVALID 반환
    return "INVALID"

@app.get("/", response_class=HTMLResponse)
async def read_root():
    template_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()

@app.get("/analyze")
async def analyze(ticker_query: str):
    try:
        # 1. GPT를 통해 가공된 티커를 받아옵니다.
        ticker = resolve_ticker(ticker_query)

        # 유효하지 않은 입력(스포츠팀 등) 차단
        if ticker == "INVALID" or not ticker or len(ticker) > 15:
            return {
                "error": f"'{ticker_query}'은(는) 분석 가능한 종목으로 찾을 수 없거나 유효하지 않은 입력입니다. (스포츠팀/인물명 등 제외)",
                "status": "INVALID_INPUT"
            }

        # 2. yfinance 데이터 다운로드
        try:
            df = fetch_stock_data(ticker)
        except Exception as e:
            return {
                "error": f"'{ticker}'에 대한 주가 데이터를 찾을 수 없습니다. (상장폐지 또는 오매핑)",
                "status": "NO_DATA"
            }

        if df.empty:
            return {
                "error": f"'{ticker}'의 데이터셋이 비어있어 분석이 불가능합니다.",
                "status": "EMPTY_DATA"
            }

        # 3. 분석 로직
        company = fetch_company_info(ticker)
        df = calculate_indicators(df)
        signal, score = generate_signal(df)
        rsi = df["RSI"].iloc[-1]
        macd = df["MACD"].iloc[-1]

        # MACD 강도 비율 계산
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
            "rule_style": rule_style,
            "status": "SUCCESS"
        }
    except Exception as e:
        print(f"Error analyzing ticker {ticker_query}: {str(e)}")
        return {
            "error": f"데이터 처리 중 에러 발생: {str(e)}",
            "status": "ERROR"
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
