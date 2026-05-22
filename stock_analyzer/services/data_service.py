# -*- coding: utf-8 -*-
import requests
import pandas as pd
import yfinance as yf
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
    if not NEWS_API_KEY:
        return []

    try:
        from datetime import datetime, timedelta
        # 최근 1개월(약 30일) 전 날짜 계산
        one_month_ago = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        url = "https://newsapi.org/v2/everything"
        
        # 1차 시도: 티커로 검색
        query = f'"{ticker}"'
        if company_name:
            # 2차 시도: 회사명 포함 검색
            query = f'"{ticker}" OR "{company_name}"'

        params = {
            "q": query,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 40,
            "from": one_month_ago, # 최근 1개월 데이터 요청
            "apiKey": NEWS_API_KEY
        }

        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, params=params, headers=headers, timeout=10)
        data = r.json()

        articles = data.get("articles", [])
        
        # 만약 결과가 너무 적으면 회사명으로만 재검색 (Broad search)
        if len(articles) < 3 and company_name:
            params["q"] = company_name
            r = requests.get(url, params=params, headers=headers, timeout=10)
            articles = r.json().get("articles", [])

        items = []
        seen_titles = set()

        for a in articles:
            title = a.get("title", "")
            if not title or title in seen_titles:
                continue
            
            # 필터링 조건 완화: 티커나 회사명 중 하나라도 제목에 포함되면 인정
            # 단, 너무 일반적인 단어일 경우를 대비해 간단한 체크
            lower_title = title.lower()
            ticker_match = ticker.lower() in lower_title
            company_match = (
                any(word.lower() in lower_title for word in company_name.split())
                if company_name else False
            )

            if not (ticker_match or company_match):
                # 제목에 없더라도 설명(description)에 있을 수 있음
                desc = a.get("description", "") or ""
                if not (ticker.lower() in desc.lower()):
                    continue

            items.append(
                NewsItem(
                    title=title,
                    url=a.get("url", "")
                )
            )
            seen_titles.add(title)
            if len(items) >= 5:
                break
                
        return items

    except Exception as e:
        print("뉴스 수집 오류:", e)
        return []
