# -*- coding: utf-8 -*-
import requests
import pandas as pd
import yfinance as yf
from pygooglenews import GoogleNews
from stock_analyzer.config import NEWS_API_KEY
from stock_analyzer.models import NewsItem

def validate_ticker(ticker: str) -> bool:
    """해당 티커가 실제로 주가 조회가 가능하고 정상적인 데이터셋을 가졌는지 검증합니다."""
    try:
        # 매우 짧은 기간(1일)만 시도하여 속도 최적화
        df = yf.download(ticker, period="1d", progress=False)
        return not df.empty
    except:
        return False

def fetch_stock_data(ticker):
    df = yf.download(
        ticker,
        period="1y",
        interval="1d",
        progress=False
    )

    if df.empty:
        raise Exception("주가 데이터 없음")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df

def fetch_company_info(ticker):
    try:
        info = yf.Ticker(ticker).info
        return {
            "name": info.get("longName", ticker),
            "sector": info.get("sector", ""),
            "industry": info.get("industry", ""),
            "marketCap": info.get("marketCap", 0),
            "quoteType": info.get("quoteType", ""),
        }
    except:
        return {}

def fetch_financials(ticker):
    try:
        stock = yf.Ticker(ticker)
        # 손익계산서 (연간)
        income_stmt = stock.income_stmt
        
        # 최근 3~4년 데이터 추출
        financials = []
        if income_stmt is not None and not income_stmt.empty:
            cols = income_stmt.columns[:4] # 최근 4년치
            for col in cols:
                date_str = str(col.date())
                revenue = income_stmt.loc["Total Revenue", col] if "Total Revenue" in income_stmt.index else 0
                op_income = income_stmt.loc["Operating Income", col] if "Operating Income" in income_stmt.index else 0
                financials.append({
                    "year": date_str,
                    "revenue": revenue,
                    "op_income": op_income
                })
        
        # 어닝 일정 및 결과 (최근)
        earnings_dates = stock.earnings_dates
        earnings_info = ""
        if earnings_dates is not None and not earnings_dates.empty:
            latest_earning = earnings_dates.iloc[0]
            eps_est = latest_earning.get("EPS Estimate", "N/A")
            eps_act = latest_earning.get("Reported EPS", "N/A")
            surprise = latest_earning.get("Surprise(%)", "N/A")
            earnings_info = f"Latest Earning: EPS Est {eps_est}, Actual {eps_act}, Surprise {surprise}"

        return {
            "history": financials,
            "earnings_summary": earnings_info
        }
    except Exception as e:
        print(f"재무 데이터 수집 오류 ({ticker}):", e)
        return {"history": [], "earnings_summary": "N/A"}

def fetch_news(ticker, company_name=""):
    """
    구글 뉴스를 사용하여 종목 관련 최신 뉴스를 가져옵니다.
    회사명과 티커를 모두 검색 쿼리에 포함하여 검색 결과의 폭을 넓힙니다.
    """
    try:
        is_korean = ticker.endswith(".KS") or ticker.endswith(".KQ")

        # 티커에서 접미사(.KS, .KQ) 제거하여 순수 코드 또는 심볼 추출
        clean_ticker = ticker.split('.')[0] if '.' in ticker else ticker

        if is_korean:
            gn = GoogleNews(lang='ko', country='KR')
            # 한국 주식 예: "더존비즈온 OR 012510"
            search_query = f'"{company_name}" OR "{clean_ticker}"' if company_name else f'"{clean_ticker}"'
        else:
            gn = GoogleNews(lang='en', country='US')
            # 미국 주식 예: "Nvidia OR NVDA"
            search_query = f'"{company_name}" OR "{clean_ticker}"' if company_name else f'"{clean_ticker}"'

        s = gn.search(search_query)
        entries = s.get('entries', [])
        
        items = []
        seen_titles = set()

        for e in entries:
            title = e.get('title', '')
            link = e.get('link', '')

            if not title or title in seen_titles:
                continue

            items.append(
                NewsItem(
                    title=title,
                    url=link
                )
            )
            seen_titles.add(title)

            if len(items) >= 20: # AI 분석을 위해 더 많은 뉴스(20개)를 수집
                break
                
        return items

    except Exception as e:
        print(f"뉴스 수집 오류 ({ticker}):", e)
        return []
