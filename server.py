# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os
from dotenv import load_dotenv

# Import core services
from stock_analyzer.services.ticker_service import resolve_ticker
from stock_analyzer.services.data_service import fetch_stock_data, fetch_company_info, fetch_news, fetch_financials
from stock_analyzer.services.analysis_service import calculate_indicators, generate_signal, calculate_probability, determine_style
from stock_analyzer.services.ai_service import analyze_ai
from stock_analyzer.services.chart_service import generate_chart

load_dotenv()

app = FastAPI(title="Stock Analyzer API")

class AnalysisRequest(BaseModel):
    ticker_query: str

@app.get("/analyze")
async def analyze(ticker_query: str):
    try:
        # 1. 티커 확인
        ticker = resolve_ticker(ticker_query)
        
        # 2. 데이터 수집
        df = fetch_stock_data(ticker)
        company = fetch_company_info(ticker)
        
        # 3. 분석
        df = calculate_indicators(df)
        signal, score = generate_signal(df)
        rsi = df["RSI"].iloc[-1]
        macd = df["MACD"].iloc[-1]
        prob = calculate_probability(score, rsi, macd, df)
        
        company_name = company.get('name', '')
        news = fetch_news(ticker, company_name)
        financials = fetch_financials(ticker)
        
        # 4. AI 분석
        ai_result = analyze_ai(ticker, signal, score, rsi, macd, prob, news, financials)
        rule_style = determine_style(company, df, rsi)
        
        # 5. 결과 조합
        return {
            "ticker": ticker,
            "company": company,
            "signal": signal,
            "score": score,
            "rsi": float(rsi),
            "macd": float(macd),
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
