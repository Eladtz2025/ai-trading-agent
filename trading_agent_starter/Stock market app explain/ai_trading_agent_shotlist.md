# Shot List – AI Trading Agent Explainer (Hebrew)

**משך:** ~3:00 דקות • **רזולוציה מומלצת:** 1920×1080 • **FPS:** 60  
**כלי הקלטה:** OBS Studio או כל מקליט מסך • **קלט אודיו:** מיקרופון דיבור

## 1) פתיח (0:00–0:06)
- מסך מלא של הדשבורד עם הכותרת “AI Trading Agent – Paper Trading”.
- זום קל פנימה בזמן הדיבור.

## 2) מה האפליקציה עושה (0:06–0:16)
- פאן איטי על כל המסך.
- הדגש עם העכבר את המשפט “Plain English Recommendation”.

## 3) Paper Trading בלבד (0:16–0:26)
- הדגש את הבאנר הצהוב “Simulation only – no real paper order” כשה-DRY_RUN דולק.
- הצג מתג DRY_RUN בסרגל השמאלי.

## 4) הסבר על Inputs (0:26–0:40)
- סמן עם העכבר: Ticker → SMA Fast → SMA Slow → Order Budget → DRY_RUN.
- עצור קל על כל שדה 1–2 שניות.

## 5) Get Recommendation (0:40–0:55)
- לחץ על הכפתור “Get Recommendation”.
- גלול אל שורת התקציר והטקסט הכחול.

## 6) פירוט Action/Amount/Stop/Take/Confidence (0:55–1:15)
- היילייט לכל פריט בשורת התקציר בתורו.

## 7) מדדים (1:15–1:35)
- הצבע על Price, SMA, ATR, RSI (ארבעת ה-Metric tiles).
- אפשר זום קצר על כל אחד.

## 8) Place order now – Bracket (1:35–1:55)
- הדגש את הכפתור “Place order now (Paper)”.
- אם DRY_RUN כבוי, הראה התראה ירוקה עם Order ID.

## 9) Open Positions (1:55–2:15)
- עבור לטבלה הימנית. אם אין פוזיציה—הצג “No open positions”. אם יש—הדגש את העמודות Qty/Avg Entry/P&L.

## 10) Open Orders (2:15–2:35)
- גלול לפנל “Open Orders (Paper)”. הדגש סטטוס NEW/HELD כשיש הזמנות ממתינות.

## 11) Close ALL (2:35–2:50)
- לחץ על “Close ALL paper positions” (רק אם זה בסדר לסגור). הצג הצלחה.

## 12) לוגים ו-CLI (2:50–3:10)
- הצג חלון PowerShell קצר שמריץ `python src\utils_peek_duck.py` ומדפיס שורות.

## 13) סיום (3:10–3:25)
- חזור לכותרת. כתובית תחתונה: “Paper only. For learning purposes.”
