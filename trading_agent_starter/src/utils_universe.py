# src/utils_universe.py
# Small, static universes to screen quickly (can be expanded later).

DOW30 = [
    "AAPL","AMGN","AXP","BA","CAT","CRM","CSCO","CVX","DIS","DOW","GS","HD","HON",
    "IBM","INTC","JNJ","JPM","KO","MCD","MMM","MRK","MSFT","NKE","PG","TRV","UNH",
    "V","VZ","WBA","WMT"
]

NASDAQ100 = [
    "AAPL","MSFT","NVDA","GOOGL","GOOG","AMZN","META","AVGO","COST","PEP","ADBE",
    "TSLA","NFLX","AMD","INTC","CSCO","QCOM","TXN","AMAT","ADI","PANW","PDD","BKNG",
    "VRTX","SBUX","REGN","HON","MU","LRCX","GILD","MDLZ","ABNB","MAR","CRWD","KLAC",
    "ISRG","MRVL","ASML","SNPS","PYPL","FTNT","ADP","CDNS","ORLY","NXPI","KDP","MELI",
    "CTAS","KHC","AEP","ODFL","ROP","PCAR","IDXX","WDAY","ODFL","CHTR","TEAM","MNST",
    "CSX","PAYX","PANW","AMD","VRTX","AZN","MRNA","LCID","COIN","ROST","EXC","CEG",
    "CPRT","ADSK","INTU","LULU","CRWD","FAST","CTSH","KDP","ATVI","BIIB","EA","FISV",
    "JD","ZM","DOCU","ZS","MDB","DDOG","SPLK","OKTA","NTES","BIDU","KDP","GFS","ANSS"
]

SP100_LITE = [
    "AAPL","MSFT","AMZN","NVDA","GOOGL","META","BRK.B","XOM","UNH","JNJ","JPM",
    "V","PG","MA","AVGO","HD","CVX","MRK","PEP","LLY","ABBV","KO","COST","BAC",
    "WMT","DIS","MCD","CSCO","ADBE","TMO","PFE","ACN","DHR","LIN","TXN","NFLX",
    "AMD","NEE","PM","IBM","INTC","QCOM","HON","BA","CAT","GE","AMGN","LOW","GS",
    "RTX","SCHW","NOW","BKNG","SPGI","BLK","PLD","LRCX","ISRG","ETN","MDT"
]

UNIVERSES = {
    "Dow 30 (fast)": DOW30,
    "Nasdaq 100 (wide)": list(dict.fromkeys(NASDAQ100)),  # de-dupe just in case
    "S&P 100 (lite)": SP100_LITE
}
