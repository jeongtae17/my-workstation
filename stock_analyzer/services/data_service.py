# -*- coding: utf-8 -*-
import requests
import pandas as pd
import yfinance as yf
from pygooglenews import GoogleNews
from stock_analyzer.config import NEWS_API_KEY, client
from stock_analyzer.models import NewsItem


def validate_ticker(ticker: str) -> bool:
    """Check whether the given ticker returns valid price data."""
    try:
        df = yf.download(ticker, period="1d", progress=False)
        return not df.empty
    except Exception:
        return False


def fetch_stock_data(ticker):
    df = yf.download(
        ticker,
        period="1y",
        interval="1d",
        progress=False,
    )

    if df.empty:
        raise Exception("No stock data available")

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
    except Exception:
        return {}


def fetch_financials(ticker):
    try:
        stock = yf.Ticker(ticker)
        income_stmt = stock.income_stmt
        financials = []

        if income_stmt is not None and not income_stmt.empty:
            import numpy as np
            cols = income_stmt.columns[:4]
            for col in cols:
                date_str = str(col.date())
                raw_revenue = income_stmt.loc["Total Revenue", col] if "Total Revenue" in income_stmt.index else 0
                raw_op_income = income_stmt.loc["Operating Income", col] if "Operating Income" in income_stmt.index else 0
                revenue = 0 if (raw_revenue is None or (isinstance(raw_revenue, float) and np.isnan(raw_revenue))) else raw_revenue
                op_income = 0 if (raw_op_income is None or (isinstance(raw_op_income, float) and np.isnan(raw_op_income))) else raw_op_income
                financials.append({"year": date_str, "revenue": float(revenue), "op_income": float(op_income)})

        earnings_dates = stock.earnings_dates
        earnings_info = ""
        if earnings_dates is not None and not earnings_dates.empty:
            latest_earning = earnings_dates.iloc[0]
            eps_est = latest_earning.get("EPS Estimate", "N/A")
            eps_act = latest_earning.get("Reported EPS", "N/A")
            surprise = latest_earning.get("Surprise(%)", "N/A")
            earnings_info = f"Latest Earning: EPS Est {eps_est}, Actual {eps_act}, Surprise {surprise}"

        return {"history": financials, "earnings_summary": earnings_info}
    except Exception as e:
        print(f"Financial data error ({ticker}): {e}")
        return {"history": [], "earnings_summary": "N/A"}


def is_news_relevant_ai(ticker, company_name, title, description=""):
    """AI를 사용해 뉴스 관련도를 필터링합니다."""
    try:
        context = f"Company: {company_name}, Ticker: {ticker}, Title: {title}"
        if description:
            context += f", Description: {description}"
        rsp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a news relevance filter. Return only RELEVANT or NOT_RELEVANT. "
                        "Filter out false positives like BB gun for BlackBerry stock."
                    ),
                },
                {"role": "user", "content": f"Is this news relevant to {company_name}? {title}"},
            ],
            temperature=0,
            timeout=5,
        )
        result = rsp.choices[0].message.content.strip().upper()
        return "RELEVANT" in result
    except Exception as e:
        print(f"AI news filter error ({ticker}): {e}")
        text = (title + " " + description).lower()
        return company_name.lower() in text or ticker.lower() in text


def fetch_news_from_newsapi(ticker, company_name="", is_korean=False):
    if not NEWS_API_KEY:
        return []
    try:
        clean_ticker = ticker if ticker.startswith("^") else (ticker.split(".")[0] if "." in ticker else ticker)
        if ticker == "^KS11":
            query = "코스피"
        elif ticker == "^KQ11":
            query = "코스닥"
        elif ticker == "^GSPC":
            query = "S&P 500"
        elif ticker == "^IXIC":
            query = "Nasdaq Composite"
        else:
            if company_name:
                query = f'"{company_name}" OR "{clean_ticker}" stock' if not is_korean else f'"{company_name}" OR "{clean_ticker}"'
            else:
                query = f'"{clean_ticker}"'
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "sortBy": "publishedAt",
            "language": "ko" if is_korean else "en",
            "pageSize": 30,
            "apiKey": NEWS_API_KEY,
        }
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        if data.get("status") != "ok":
            return []
        items = []
        seen_titles = set()
        for article in data.get("articles", []):
            title = article.get("title", "")
            url = article.get("url", "")
            description = article.get("description", "")
            if not title or title in seen_titles:
                continue
            if not is_news_relevant_ai(ticker, company_name, title, description):
                continue
            items.append(NewsItem(title=title, url=url))
            seen_titles.add(title)
            if len(items) >= 5:
                break
        return items
    except Exception as e:
        print(f"NewsAPI error ({ticker}): {e}")
        return []


def fetch_news(ticker, company_name=""):
    try:
        is_korean = ticker.endswith(".KS") or ticker.endswith(".KQ") or ticker == "^KS11" or ticker == "^KQ11"
        items = fetch_news_from_newsapi(ticker, company_name, is_korean)
        if items:
            return items

        try:
            clean_ticker = ticker if ticker.startswith("^") else (ticker.split(".")[0] if "." in ticker else ticker)
            gn = GoogleNews(lang="ko" if is_korean else "en", country="KR" if is_korean else "US")
            if ticker == "^KS11":
                search_query = "코스피 지수"
            elif ticker == "^KQ11":
                search_query = "코스닥 지수"
            elif ticker == "^GSPC":
                search_query = "S&P 500 index"
            elif ticker == "^IXIC":
                search_query = "Nasdaq Composite index"
            else:
                if ticker.upper() == "BB":
                    search_query = f'"{company_name}" OR "{clean_ticker}" -"BB gun" -"toy gun" -airsoft -paintball -"school bus"'
                else:
                    search_query = f'"{company_name}" OR "{clean_ticker}"' if company_name else f'"{clean_ticker}"'
            s = gn.search(search_query)
            items = []
            seen_titles = set()
            for e in s.get("entries", [])[:20]:
                title = e.get("title", "")
                link = e.get("link", "")
                if not title or title in seen_titles:
                    continue
                items.append(NewsItem(title=title, url=link))
                seen_titles.add(title)
                if len(items) >= 20:
                    break
            return items
        except Exception as e:
            print(f"Google News error ({ticker}): {e}")
            return []
    except Exception as e:
        print(f"News fetch error ({ticker}): {e}")
        return []
