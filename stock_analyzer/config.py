# -*- coding: utf-8 -*-
import os
from dotenv import load_dotenv
from openai import OpenAI

# 환경 변수 로드
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=OPENAI_API_KEY)

# GUI 관련 설정 (서버 환경에서는 import 에러 방지)
try:
    import customtkinter as ctk
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    
    PRIMARY_BG = "#0B1120"
    CARD_BG    = "#172033"
    CARD_2     = "#1E293B"
    ACCENT     = "#3B82F6"
    SUCCESS    = "#22C55E"
    WARNING    = "#F59E0B"
    DANGER     = "#EF4444"
    TEXT       = "#F8FAFC"
    SUBTEXT    = "#94A3B8"

    TITLE_FONT = ("Malgun Gothic", 28, "bold")
    HEADER_FONT = ("Malgun Gothic", 18, "bold")
    BODY_FONT = ("Malgun Gothic", 14)
    SMALL_FONT = ("Malgun Gothic", 12)
except ImportError:
    # 서버 환경용 더미 상수
    PRIMARY_BG = CARD_BG = CARD_2 = ACCENT = SUCCESS = WARNING = DANGER = TEXT = SUBTEXT = ""
    TITLE_FONT = HEADER_FONT = BODY_FONT = SMALL_FONT = None
