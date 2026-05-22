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

# 종목명 -> 티커 매핑 (정확한 티커로 수정)
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

SUBSIDIARY_KEYWORD_MAP = {
    "HANWHAENGINE": "082740.KS",
    "HANWHAGENERALINSURANCE": "000370.KS",
    "HANWHAINSURANCE": "000370.KS",
    "HANWHA손해보험": "000370.KS"
}

def get_ticker_by_keyword(normalized_name: str):
    if "HANWHA" in normalized_name and "ENGINE" in normalized_name:
        return "082740.KS"
    if "HANWHA" in normalized_name and ("손해" in normalized_name or "손보" in normalized_name) and "보험" in normalized_name:
        return "000370.KS"
    for keyword, ticker in SUBSIDIARY_KEYWORD_MAP.items():
        if keyword in normalized_name:
            return ticker
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
        # 불필요한 따옴표, 공백, 보이지 않는 문자 전처리 강화
        ticker = "".join(c for c in ticker if c.isalnum() or c in ".-").strip()

        if not ticker or len(ticker) > 12:
            return None

        return ticker
    except:
        return None

def resolve_ticker(user_input: str):
    user_input = user_input.strip()
    normalized_input = normalize_ticker_key(user_input)

    # 1. 하드코딩 맵 우선 확인
    if user_input in KOREAN_TICKER_MAP:
        return KOREAN_TICKER_MAP[user_input]
    if user_input.upper() in KOREAN_TICKER_MAP:
        return KOREAN_TICKER_MAP[user_input.upper()]
    if normalized_input in NORMALIZED_TICKER_MAP:
        return NORMALIZED_TICKER_MAP[normalized_input]

    # 2. 보조 키워드 매칭으로 자회사/계열사 대응
    subsidiary_ticker = get_ticker_by_keyword(normalized_input)
    if subsidiary_ticker:
        return subsidiary_ticker

    # 3. GPT 검색 + 유효성 검증
    gpt_ticker = resolve_ticker_with_gpt(user_input)
    if gpt_ticker and validate_ticker(gpt_ticker):
        return gpt_ticker

    # 4. yfinance 검색으로 추가 후보 찾기
    searched_ticker = search_ticker_by_name(user_input)
    if searched_ticker:
        return searched_ticker

    # 5. GPT가 실패하거나 유효하지 않은 경우 마지막 수단으로 입력값 그대로 시도
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