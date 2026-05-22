# -*- coding: utf-8 -*-
import requests
import pandas as pd
import yfinance as yf
from pygooglenews import GoogleNews
from stock_analyzer.config import NEWS_API_KEY, client
from stock_analyzer.models import NewsItem

def validate_ticker(ticker: str) -> bool:
    """?•´?‹¹ ?‹°ì»¤ê?? ?‹¤? œë¡? ì£¼ê?? ì¡°íšŒê°? ê°??Š¥?•˜ê³? ? •?ƒ? ?¸ ?°?´?„°?…‹?„ ê°?ì¡ŒëŠ”ì§? ê²?ì¦í•©?‹ˆ?‹¤."""
    try:
        # ë§¤ìš° ì§§ì?? ê¸°ê°„(1?¼)ë§? ?‹œ?„?•˜?—¬ ?†?„ ìµœì ?™”
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
        raise Exception("ì£¼ê?? ?°?´?„° ?—†?Œ")

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
        # ?†?µê³„ì‚°?„œ (?—°ê°?)
        income_stmt = stock.income_stmt
        
        # ìµœê·¼ 3~4?…„ ?°?´?„° ì¶”ì¶œ
        financials = []
        if income_stmt is not None and not income_stmt.empty:
            import numpy as np # NaN ì²˜ë¦¬ë¥? ?œ„?•´ ì¶”ê??
            cols = income_stmt.columns[:4] # ìµœê·¼ 4?…„ì¹?
            for col in cols:
                date_str = str(col.date())

                # ?°?´?„°ê°? ?—†ê±°ë‚˜ NaN?¸ ê²½ìš° 0?œ¼ë¡? ì²˜ë¦¬?•˜?—¬ JSON ?—?Ÿ¬ ë°©ì??
                raw_revenue = income_stmt.loc["Total Revenue", col] if "Total Revenue" in income_stmt.index else 0
                raw_op_income = income_stmt.loc["Operating Income", col] if "Operating Income" in income_stmt.index else 0

                revenue = 0 if (raw_revenue is None or (isinstance(raw_revenue, float) and np.isnan(raw_revenue))) else raw_revenue
                op_income = 0 if (raw_op_income is None or (isinstance(raw_op_income, float) and np.isnan(raw_op_income))) else raw_op_income

                financials.append({
                    "year": date_str,
                    "revenue": float(revenue), # JSON ?˜¸?™˜?„ ?œ„?•´ ëª…ì‹œ?  float ë³??™˜
                    "op_income": float(op_income)
                })
        
        # ?–´?‹ ?¼? • ë°? ê²°ê³¼ (ìµœê·¼)
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
        print(f"?¬ë¬? ?°?´?„° ?ˆ˜ì§? ?˜¤ë¥? ({ticker}):", e)
        return {"history": [], "earnings_summary": "N/A"}

def fetch_news(ticker, company_name=""):
    """
    êµ¬ê?? ?‰´?Š¤ë¥? ?‚¬?š©?•˜?—¬ ì¢…ëª© ê´?? ¨ ìµœì‹  ?‰´?Š¤ë¥? ê°?? ¸?˜µ?‹ˆ?‹¤.
    ?šŒ?‚¬ëª…ê³¼ ?‹°ì»¤ë?? ëª¨ë‘ ê²??ƒ‰ ì¿¼ë¦¬?— ?¬?•¨?•˜?—¬ ê²??ƒ‰ ê²°ê³¼?˜ ?­?„ ?„“?™?‹ˆ?‹¤.
    """
    try:
        is_korean = ticker.endswith(".KS") or ticker.endswith(".KQ") or ticker == "^KS11" or ticker == "^KQ11"

        # ?‹°ì»¤ì—?„œ ? ‘ë¯¸ì‚¬(.KS, .KQ) ? œê±°í•˜?—¬ ?ˆœ?ˆ˜ ì½”ë“œ ?˜?Š” ?‹¬ë³? ì¶”ì¶œ (?¸?±?Š¤?Š” ?˜ˆ?™¸)
        if ticker.startswith('^'):
            clean_ticker = ticker
        else:
            clean_ticker = ticker.split('.')[0] if '.' in ticker else ticker

        if is_korean:
            gn = GoogleNews(lang='ko', country='KR')
            # ?•œêµ? ì£¼ì‹/ì§??ˆ˜ ?˜ˆ: "?”ì¡´ë¹„ì¦ˆì˜¨ OR 012510", "ì½”ìŠ¤?”¼ OR ^KS11"
            if ticker == "^KS11": search_query = "ì½”ìŠ¤?”¼ ì§??ˆ˜ ? „ë§?"
            elif ticker == "^KQ11": search_query = "ì½”ìŠ¤?‹¥ ì§??ˆ˜ ? „ë§?"
            else: search_query = f'"{company_name}" OR "{clean_ticker}"' if company_name else f'"{clean_ticker}"'
        else:
            gn = GoogleNews(lang='en', country='US')
            # ë¯¸êµ­ ì£¼ì‹/ì§??ˆ˜
            if ticker == "^GSPC": search_query = "S&P 500 index"
            elif ticker == "^IXIC": search_query = "Nasdaq Composite index"
            else: search_query = f'"{company_name}" OR "{clean_ticker}"' if company_name else f'"{clean_ticker}"'

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

            if len(items) >= 20: # AI ë¶„ì„?„ ?œ„?•´ ?” ë§ì?? ?‰´?Š¤(20ê°?)ë¥? ?ˆ˜ì§?
                break
                
        return items

    except Exception as e:
        print(f"?‰´?Š¤ ?ˆ˜ì§? ?˜¤ë¥? ({ticker}):", e)
        return []
