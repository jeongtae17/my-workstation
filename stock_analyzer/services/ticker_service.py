# -*- coding: utf-8 -*-
from stock_analyzer.config import client

def resolve_ticker(user_input: str):
    """
    GPT-4o를 사용하여 사용자의 자연어 입력(한글, 영어, 회사명 등)을 
    최적의 Yahoo Finance 티커로 변환합니다.
    """
    user_input = user_input.strip()

    # 1. 1-5자 알파벳은 티커로 간주하여 즉시 대문자로 변환 (dxyz -> DXYZ)
    if 1 <= len(user_input) <= 5 and user_input.isalpha():
        return user_input.upper()

    try:
        rsp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "당신은 글로벌 금융 시장의 모든 상장 자산(주식, ETF, 인덱스)을 꿰뚫고 있는 수석 데이터 엔지니어입니다.\n\n"
                        "### 임무:\n"
                        "사용자가 입력한 검색어(특히 길고 복잡한 한글 회사명)를 분석하여 Yahoo Finance에서 데이터 조회가 가능한 '가장 정확한 티커 심볼' 하나만 도출하십시오.\n\n"
                        "### 검색 고도화 가이드라인:\n"
                        "1. **의미적 매칭(Semantic Matching)**: 입력어의 핵심 키워드를 추출하여 공식 명칭을 추론하십시오.\n"
                        "   - 예: '데스티니 테크 100' -> 'Destiny Tech100'임을 인지하고 티커 'DXYZ' 도출.\n"
                        "   - 예: '로켓 랩' -> 'Rocket Lab'임을 인지하고 티커 'RKLB' 도출.\n"
                        "2. **유사어 처리**: 사용자가 띄어쓰기를 틀리거나 줄여 말해도(예: '삼전', '엔비') 원래의 기업을 정확히 찾아야 합니다.\n"
                        "3. **ETF 특화**: 이름에 '100', '500', '레버리지', '인버스' 등이 포함되면 해당 상품의 티커를 우선합니다.\n"
                        "4. **한국 시장 예외**: 한국 기업은 반드시 접미사(.KS 또는 .KQ)를 유지하십시오.\n"
                        "   - '삼성전자' -> '005930.KS', '카카오' -> '035720.KS'\n\n"
                        "### 출력 규칙 (절대 준수):\n"
                        "- **오직 티커 심볼 문자열만 출력하십시오.** (예: NVDA)\n"
                        "- 어떤 설명, 접두어(TICKER:), 마침표, 따옴표도 포함하지 마십시오.\n"
                        "- 모든 영문은 대문자로 출력하십시오.\n\n"
                        "### 추론 예시:\n"
                        "- '데스티니 테크 100' -> 'DXYZ'\n"
                        "- '로켓 랩' -> 'RKLB'\n"
                        "- '아이쉐어즈 글로벌 100' -> 'IOO'\n"
                        "- '엔비디아' -> 'NVDA'\n"
                        "- '미국 테크 TOP10' -> 'QQQM'\n"
                        "- '애플' -> 'AAPL'"
                    )
                },
                {
                    "role": "user",
                    "content": f"Resolve this term to a Ticker: {user_input}"
                }
            ],
            temperature=0,
            max_tokens=15
        )

        ticker = rsp.choices[0].message.content.strip().upper()
        
        # 클리닝 로직: 혹시 모를 불필요한 텍스트 제거
        ticker = ticker.split()[-1] if " " in ticker else ticker
        ticker = ticker.split(":")[-1].strip()
        ticker = "".join(c for c in ticker if c.isalnum() or c in ".-")
        
        return ticker

    except Exception as e:
        raise Exception(f"Ticker Resolution Logic Error: {str(e)}")
