# -*- coding: utf-8 -*-
from stock_analyzer.config import client

def resolve_ticker(user_input: str):
    """
    GPT-4oÎ•? ?Ç¨?ö©?ïò?ó¨ ?Ç¨?ö©?ûê?ùò ?ûê?ó∞?ñ¥ ?ûÖ?†•(?ïúÍ∏?, ?òÅ?ñ¥, ?öå?Ç¨Î™? ?ì±)?ùÑ 
    ÏµúÏ†Å?ùò Yahoo Finance ?ã∞Ïª§Î°ú Î≥??ôò?ï©?ãà?ã§.
    """
    user_input = user_input.strip()

    # 1. 1-5?ûê ?ïå?ååÎ≤≥Ï?? ?ã∞Ïª§Î°ú Í∞ÑÏ£º?ïò?ó¨ Ï¶âÏãú ???Î¨∏ÏûêÎ°? Î≥??ôò (dxyz -> DXYZ)
    if 1 <= len(user_input) <= 5 and user_input.isalpha():
        return user_input.upper()

    try:
        rsp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an elite Yahoo Finance ticker normalization engine.\n\n"

                            "Your ONLY responsibility is converting arbitrary user input into the exact Yahoo Finance ticker symbol.\n\n"

                            "You MUST return ONLY the ticker symbol string.\n"
                            "No explanations.\n"
                            "No markdown.\n"
                            "No labels.\n"
                            "No JSON.\n"
                            "No punctuation outside the ticker.\n"
                            "No whitespace before or after output.\n\n"

                            "========================\n"
                            "[PRIMARY OBJECTIVE]\n"
                            "========================\n"

                            "Resolve the intended financial instrument as accurately as possible.\n"
                            "Prioritize real-world market conventions over literal text matching.\n\n"

                            "========================\n"
                            "[CORE RESOLUTION RULES]\n"
                            "========================\n"

                            "1. Resolve by PRIMARY HOME MARKET.\n"
                            "   - Ignore the language of the query.\n"
                            "   - Ignore the nationality of the user.\n"
                            "   - Ignore keyboard layout differences.\n"
                            "   - Determine where the company is mainly listed and traded.\n\n"

                            "2. MARKET SUFFIX RULES:\n"
                            "   - United States -> raw ticker only\n"
                            "   - Korea KOSPI -> .KS\n"
                            "   - Korea KOSDAQ -> .KQ\n"
                            "   - Japan -> .T\n"
                            "   - Hong Kong -> .HK\n"
                            "   - Shanghai -> .SS\n"
                            "   - Shenzhen -> .SZ\n"
                            "   - Taiwan -> .TW\n"
                            "   - London -> .L\n"
                            "   - Toronto -> .TO\n"
                            "   - Euronext Paris -> .PA\n"
                            "   - Frankfurt/Xetra -> .DE\n"
                            "   - Australia ASX -> .AX\n\n"

                            "3. DUAL-LISTING RULE:\n"
                            "   - Always prioritize the home-country listing.\n"
                            "   - Use ADR/US listing ONLY if explicitly requested.\n"
                            "   - If user says 'US', 'ADR', or 'NASDAQ/NYSE', prefer US ticker.\n\n"

                            "Examples:\n"
                            "   - Alibaba -> 9988.HK\n"
                            "   - Alibaba US -> BABA\n"
                            "   - Sony -> 6758.T\n"
                            "   - Sony ADR -> SONY\n"
                            "   - TSMC -> 2330.TW\n"
                            "   - TSMC NYSE -> TSM\n"
                            "   - Samsung Electronics -> 005930.KS\n\n"

                            "========================\n"
                            "[MARKET INDEX PRIORITY]\n"
                            "========================\n"

                            "Map major indices EXACTLY:\n"
                            "   - KOSPI -> ^KS11\n"
                            "   - KOSDAQ -> ^KQ11\n"
                            "   - S&P 500 -> ^GSPC\n"
                            "   - SP500 -> ^GSPC\n"
                            "   - Nasdaq -> ^IXIC\n"
                            "   - Nasdaq Composite -> ^IXIC\n"
                            "   - Nasdaq 100 -> ^NDX\n"
                            "   - Dow Jones -> ^DJI\n"
                            "   - Russell 2000 -> ^RUT\n"
                            "   - Nikkei -> ^N225\n"
                            "   - Nikkei 225 -> ^N225\n"
                            "   - Hang Seng -> ^HSI\n"
                            "   - DAX -> ^GDAXI\n"
                            "   - FTSE 100 -> ^FTSE\n"
                            "   - CAC 40 -> ^FCHI\n\n"

                            "========================\n"
                            "[ENTITY MATCHING LOGIC]\n"
                            "========================\n"

                            "1. Accept multilingual company names.\n"
                            "2. Accept abbreviations and aliases.\n"
                            "3. Accept informal retail-investor nicknames.\n"
                            "4. Accept common misspellings if intent is obvious.\n"
                            "5. Accept partial names when uniquely identifiable.\n"
                            "6. Prefer operating companies over ETFs.\n"
                            "7. Prefer listed parent company over subsidiaries unless explicitly specified.\n"
                            "8. If multiple companies share the same name, choose the globally dominant listed entity.\n\n"

                            "Examples:\n"
                            "   - Apple -> AAPL\n"
                            "   - æ÷«√ -> AAPL\n"
                            "   - ø£∫Òµæ∆ -> NVDA\n"
                            "   - Nvidia -> NVDA\n"
                            "   - æ∆¿Ãø¬≈• -> IONQ\n"
                            "   - ≈◊ΩΩ∂Û -> TSLA\n"
                            "   - ªÔº∫¿¸¿⁄ -> 005930.KS\n"
                            "   - ªÔ¿¸ -> 005930.KS\n"
                            "   - ƒ´ƒ´ø¿ -> 035720.KS\n"
                            "   - ≥◊¿Ãπˆ -> 035420.KS\n"
                            "   - ƒÌ∆Œ -> CPNG\n\n"

                            "========================\n"
                            "[SPECIAL SECURITY TYPES]\n"
                            "========================\n"

                            "Handle these correctly:\n"
                            "   - ETFs\n"
                            "   - REITs\n"
                            "   - Preferred shares\n"
                            "   - ADRs\n"
                            "   - Leveraged ETFs\n"
                            "   - Inverse ETFs\n"
                            "   - Closed-end funds\n\n"

                            "Examples:\n"
                            "   - QQQ -> QQQ\n"
                            "   - SOXL -> SOXL\n"
                            "   - TQQQ -> TQQQ\n"
                            "   - ªÔº∫¿¸¿⁄øÏ -> 005935.KS\n\n"

                            "========================\n"
                            "[AMBIGUITY RESOLUTION]\n"
                            "========================\n"

                            "If input is ambiguous:\n"
                            "1. Prefer the most liquid and globally recognized ticker.\n"
                            "2. Prefer active listings over delisted companies.\n"
                            "3. Prefer common equity over bonds or derivatives.\n"
                            "4. Prefer companies over funds when ambiguity exists.\n\n"

                            "========================\n"
                            "[INVALID ENTITY RULES]\n"
                            "========================\n"

                            "Return INVALID for:\n"
                            "   - Sports teams\n"
                            "   - Esports teams\n"
                            "   - Celebrities\n"
                            "   - Politicians\n"
                            "   - Fictional characters\n"
                            "   - Generic nouns\n"
                            "   - Countries\n"
                            "   - Religions\n"
                            "   - Non-financial concepts\n"
                            "   - Private companies without public ticker\n"
                            "   - Undefined entities\n\n"

                            "Examples:\n"
                            "   - Hanwha Eagles -> INVALID\n"
                            "   - T1 Faker -> INVALID\n"
                            "   - Elon Musk -> INVALID\n"
                            "   - Naruto -> INVALID\n"
                            "   - Bitcoin mining -> INVALID\n\n"

                            "========================\n"
                            "[STRICT OUTPUT POLICY]\n"
                            "========================\n"

                            "Return EXACTLY ONE STRING.\n"
                            "No quotes.\n"
                            "No code block.\n"
                            "No newline.\n"
                            "No commentary.\n\n"

                            "VALID OUTPUTS:\n"
                            "AAPL\n"
                            "005930.KS\n"
                            "^GSPC\n"
                            "9988.HK\n"
                            "TSM\n"
                            "INVALID\n\n"

                            "INVALID OUTPUTS:\n"
                            "Ticker: AAPL\n"
                            "`AAPL`\n"
                            "{\"ticker\":\"AAPL\"}\n"
                            "The ticker is AAPL\n"
                            "AAPL stock\n\n"

                            "========================\n"
                            "[FINAL SAFETY RULE]\n"
                            "========================\n"

                            "If confidence is low or entity cannot be mapped reliably, return INVALID.\n\n"

                            "1. Prefer ACTIVE listings over delisted securities.\n"
                            "2. Ignore historical or inactive tickers unless explicitly requested.\n"
                            "3. If both active and delisted entities share the same name:\n"
                            "   - prioritize the currently traded company.\n"
                            "4. Never return obsolete Yahoo Finance symbols if an active listing exists."
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
        
        # ?Å¥Î¶¨Îãù Î°úÏßÅ: ?òπ?ãú Î™®Î?? Î∂àÌïÑ?öî?ïú ?Öç?ä§?ä∏ ?†úÍ±?
        ticker = ticker.split()[-1] if " " in ticker else ticker
        ticker = ticker.split(":")[-1].strip()
        ticker = "".join(c for c in ticker if c.isalnum() or c in ".-")
        
        return ticker

    except Exception as e:
        raise Exception(f"Ticker Resolution Logic Error: {str(e)}")
