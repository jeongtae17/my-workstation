# -*- coding: utf-8 -*-
import threading
import traceback
import customtkinter as ctk
from PIL import Image

from stock_analyzer.config import (
    PRIMARY_BG, CARD_BG, CARD_2, ACCENT, SUBTEXT,
    TITLE_FONT, HEADER_FONT, BODY_FONT, SMALL_FONT
)
from stock_analyzer.services.ticker_service import resolve_ticker
from stock_analyzer.services.data_service import fetch_stock_data, fetch_company_info, fetch_news, fetch_financials
from stock_analyzer.services.analysis_service import calculate_indicators, generate_signal, calculate_probability, determine_style
from stock_analyzer.services.ai_service import analyze_ai
from stock_analyzer.services.chart_service import generate_chart

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("AI STOCK ANALYZER PRO")
        self.geometry("1500x950")
        self.configure(fg_color=PRIMARY_BG)

        # 상단바
        top = ctk.CTkFrame(
            self,
            fg_color=CARD_BG,
            corner_radius=24,
            border_width=1,
            border_color="#334155"
        )
        top.pack(fill="x", padx=20, pady=20)

        title = ctk.CTkLabel(
            top,
            text="📈 AI STOCK ANALYZER PRO",
            font=TITLE_FONT
        )
        title.pack(side="left", padx=20, pady=20)

        self.input_var = ctk.StringVar()
        self.entry = ctk.CTkEntry(
            top,
            width=280,
            height=46,
            textvariable=self.input_var,
            placeholder_text="티커 입력 (예: NVDA)",
            corner_radius=14
        )
        self.entry.pack(side="left", padx=10)
        self.entry.bind("<Return>", lambda e: self.start_analysis())

        self.run_btn = ctk.CTkButton(
            top,
            text="분석 시작",
            width=120,
            height=46,
            fg_color=ACCENT,
            hover_color="#2563EB",
            corner_radius=14,
            command=self.start_analysis
        )
        self.run_btn.pack(side="left", padx=10)

        self.status = ctk.CTkLabel(
            top,
            text="READY",
            font=SMALL_FONT,
            text_color=SUBTEXT
        )
        self.status.pack(side="right", padx=20)

        # 메인
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        main.grid_columnconfigure(0, weight=3)
        main.grid_columnconfigure(1, weight=1)
        main.grid_rowconfigure(0, weight=2)
        main.grid_rowconfigure(1, weight=3)

        # 좌측 결과
        left = ctk.CTkFrame(main, fg_color=CARD_BG, corner_radius=24)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12), pady=(0, 12))

        self.result_box = ctk.CTkTextbox(
            left,
            font=BODY_FONT,
            fg_color=CARD_2,
            corner_radius=18
        )
        self.result_box.pack(fill="both", expand=True, padx=20, pady=20)

        # 메타 카드
        right = ctk.CTkFrame(main, fg_color=CARD_BG, corner_radius=24)
        right.grid(row=0, column=1, sticky="nsew", pady=(0, 12))

        self.meta_label = ctk.CTkLabel(
            right,
            text="분석 대기 중",
            justify="left",
            font=BODY_FONT
        )
        self.meta_label.pack(anchor="w", padx=20, pady=20)

        # 차트 카드
        chart_card = ctk.CTkFrame(main, fg_color=CARD_BG, corner_radius=24)
        chart_card.grid(row=1, column=0, columnspan=2, sticky="nsew")

        self.chart_label = ctk.CTkLabel(chart_card, text="")
        self.chart_label.pack(fill="both", expand=True, padx=20, pady=20)

    def set_status(self, text):
        self.after(0, lambda: self.status.configure(text=text))

    def write(self, text):
        def _write():
            self.result_box.insert("end", text)
            self.result_box.see("end")
        self.after(0, _write)

    def start_analysis(self):
        ticker = self.input_var.get().strip()
        if not ticker:
            return
        self.result_box.delete("1.0", "end")
        threading.Thread(
            target=self.run_analysis,
            args=(ticker,),
            daemon=True
        ).start()

    def run_analysis(self, user_input):
        try:
            self.set_status("[1/5] 티커 분석 중")
            ticker = resolve_ticker(user_input)
            self.write(f"\n📌 Ticker: {ticker}\n\n")

            self.set_status("[2/5] 데이터 수집 중")
            df = fetch_stock_data(ticker)
            company = fetch_company_info(ticker)

            self.set_status("[3/5] 기술지표 계산 중")
            df = calculate_indicators(df)
            signal, score = generate_signal(df)
            latest = df.iloc[-1]
            rsi = latest["RSI"]
            macd = latest["MACD"]
            prob = calculate_probability(score, rsi, macd, df)

            self.set_status("[4/5] 데이터/뉴스 분석 중")
            company_name = company.get('name', '')
            news = fetch_news(ticker, company_name)
            financials = fetch_financials(ticker)

            self.set_status("[5/5] AI 분석 중")
            ai = analyze_ai(ticker, signal, score, rsi, macd, prob, news, financials)
            rule_style = determine_style(company, df, rsi)
            chart_path = generate_chart(ticker, df)

            # 출력
            self.write("=" * 60 + "\n")
            self.write(f"🏢 COMPANY: {company.get('name', ticker)}\n")
            self.write(f"🏭 SECTOR: {company.get('sector', '-')}\n")
            self.write(f"📈 SIGNAL: {signal}\n")
            self.write(f"📊 SCORE: {score}\n")
            self.write(f"📉 RSI: {rsi:.2f}\n")
            self.write(f"📉 MACD: {macd:.2f}\n")
            self.write(f"🎯 AI 상승 확률: {prob}%\n\n")

            if financials and financials.get("history"):
                self.write("📊 FINANCIALS (3Y)\n")
                for h in financials["history"]:
                    self.write(f"- {h['year']}: 매출 {h['revenue']/1e9:.1f}B, 영익 {h['op_income']/1e9:.1f}B\n")
                self.write(f"📢 {financials.get('earnings_summary', '')}\n\n")

            self.write("📰 NEWS\n\n")
            for n in news:
                self.write(f"- {n.title}\n")
            self.write("\n")
            self.write("=" * 60 + "\n")
            self.write("🤖 AI ANALYSIS\n\n")
            self.write(f"{ai['summary']}\n\n")
            self.write(f"⚠️ RISK: {ai['risk']}\n")
            self.write(f"📌 AI STYLE: {ai['style']}\n")
            self.write(f"📊 RULE STYLE: {rule_style}\n")

            # 메타
            meta = f"""
📌 Ticker
{ticker}

📈 Signal
{signal}

📊 Score
{score}

📉 RSI
{rsi:.2f}

📉 MACD
{macd:.2f}

🎯 Probability
{prob}%

⚠️ Risk
{ai['risk']}

📌 AI Style
{ai['style']}

📊 Rule Style
{rule_style}
"""
            self.after(0, lambda: self.meta_label.configure(text=meta))

            # 차트 표시
            self.show_chart(chart_path)
            self.set_status("DONE")

        except Exception as e:
            self.write(f"\n❌ ERROR\n{str(e)}\n\n")
            self.write(traceback.format_exc())
            self.set_status("ERROR")

    def show_chart(self, path):
        try:
            with Image.open(path) as img:
                copied = img.copy()
            chart = ctk.CTkImage(
                light_image=copied,
                dark_image=copied,
                size=(1350, 500)
            )
            def _update():
                self.chart_label.configure(image=chart)
                self.chart_label.image = chart
            self.after(0, _update)
        except Exception as e:
            self.write(f"\n차트 표시 실패: {e}\n")
