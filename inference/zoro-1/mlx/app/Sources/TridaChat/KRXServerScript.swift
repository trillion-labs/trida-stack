// Generated from Zoro-1 Chat's KRX MCP server (Python); kept here so the app can write it out and run it with uv.
// Edit the Python and keep the two in step.

enum KRXServerScript {
    /// Written to ~/Library/Application Support/Zoro-1 Chat/mcp/krx_server.py before the "krx" server starts.
    static let source = #"""
"""Zoro-1 Chat's KRX server (MCP, stdio): KOSPI / KOSDAQ index levels and stock prices by name, from KRX via pykrx.

Two tools instead of pykrx-mcp's eight: a 4B model calls these right (a name, not a ticker; an index by name).
KRX publishes daily data, so the latest row is the last trading day's close (during market hours, today's level so
far when KRX has it). KRX's data service needs a login: pykrx reads KRX_ID / KRX_PW from the environment.
"""
from datetime import date, timedelta

from mcp.server.fastmcp import FastMCP
from pykrx import stock

INDEXES = {"KOSPI": "1001", "코스피": "1001", "KOSPI200": "1028", "코스피200": "1028",
           "KOSDAQ": "2001", "코스닥": "2001", "KOSDAQ150": "2203", "코스닥150": "2203"}
mcp = FastMCP("zoro-krx")
_names: dict = {}
# English names a model writes for the largest listings (KRX lists Korean names only)
ALIASES = {
    "samsung electronics": "삼성전자", "samsung": "삼성전자", "sk hynix": "SK하이닉스", "hynix": "SK하이닉스",
    "lg energy solution": "LG에너지솔루션", "samsung biologics": "삼성바이오로직스", "hyundai motor": "현대차",
    "hyundai": "현대차", "kia": "기아", "celltrion": "셀트리온", "kb financial": "KB금융", "shinhan": "신한지주",
    "posco holdings": "POSCO홀딩스", "posco": "POSCO홀딩스", "naver": "NAVER", "kakao": "카카오",
    "lg chem": "LG화학", "samsung sdi": "삼성SDI", "hanwha aerospace": "한화에어로스페이스",
    "hd hyundai heavy industries": "HD현대중공업", "samsung c&t": "삼성물산", "lg electronics": "LG전자",
    "sk innovation": "SK이노베이션", "sk telecom": "SK텔레콤", "kt": "KT", "hybe": "하이브",
    "ecopro bm": "에코프로비엠", "ecopro": "에코프로", "alteogen": "알테오젠",
}


def _window(days: int):
    days = max(1, min(int(days), 60))
    end = date.today()
    return (end - timedelta(days=days * 2 + 10)).strftime("%Y%m%d"), end.strftime("%Y%m%d"), days


def _failed(e: Exception) -> str:
    return (f"error: the KRX request failed ({type(e).__name__}: {e}). KRX may be down or have changed its data "
            "service, or the KRX login (KRX_ID / KRX_PW) was refused. Tell the user; don't retry with the same "
            "arguments.")


def _table(df, days: int, cols: dict) -> str:
    df = df.tail(days)
    lines = []
    prev = None
    for ix, row in df.iterrows():
        close = float(row[cols["close"]])
        change = f" ({(close / prev - 1) * 100:+.2f}%)" if prev else ""
        vol = f", volume {int(row[cols['volume']]):,}" if cols.get("volume") in row else ""
        lines.append(f"{ix.strftime('%Y-%m-%d')}: close {close:,.2f}{change}, high {float(row[cols['high']]):,.2f}, "
                     f"low {float(row[cols['low']]):,.2f}{vol}")
        prev = close
    return "\n".join(lines)


@mcp.tool(description="Korean stock market index levels (daily close, high, low) for the last few trading days. "
                      "index: KOSPI, KOSPI200, KOSDAQ or KOSDAQ150 (코스피 / 코스닥 also work).")
def index_price(index: str = "KOSPI", days: int = 5) -> str:
    code = INDEXES.get(index.strip().upper().replace(" ", "")) or INDEXES.get(index.strip().replace(" ", ""))
    if not code:
        return f"error: unknown index {index!r}; use one of KOSPI, KOSPI200, KOSDAQ, KOSDAQ150"
    start, end, days = _window(days)
    try:   # name_display=False: looking up the index's name is a separate KRX call that fails on its own
        df = stock.get_index_ohlcv_by_date(start, end, code, name_display=False)
    except Exception as e:  # noqa: BLE001
        return _failed(e)
    if df is None or df.empty:
        return "error: KRX returned no data (KRX may be unreachable, or it is a holiday)"
    return f"{index.upper()} (KRX, daily; the last row is the latest):\n" + _table(
        df, days, {"close": "종가", "high": "고가", "low": "저가", "volume": "거래량"})


def _ticker(name_or_ticker: str):
    q = name_or_ticker.strip()
    q = ALIASES.get(q.lower(), q)
    if q.isdigit() and len(q) == 6:
        return q, stock.get_market_ticker_name(q)
    if not _names:
        for market in ("KOSPI", "KOSDAQ"):
            for t in stock.get_market_ticker_list(market=market):
                _names[stock.get_market_ticker_name(t)] = t
    if q in _names:
        return _names[q], q
    hits = [n for n in _names if q.replace(" ", "").lower() in n.replace(" ", "").lower()]
    if len(hits) == 1:
        return _names[hits[0]], hits[0]
    return None, hits[:10]


@mcp.tool(description="A Korean stock's daily prices (close, high, low, volume) for the last few trading days. "
                      "stock_name: the company's Korean name as the user wrote it (e.g. 삼성전자, SK하이닉스); "
                      "a 6-digit ticker only if the user gave one.")
def stock_price(stock_name: str, days: int = 5) -> str:
    try:
        ticker, name = _ticker(stock_name)
    except Exception as e:  # noqa: BLE001
        return _failed(e)
    if ticker is None:
        if name:
            return f"error: {stock_name!r} matches several stocks: {', '.join(name)}. Use the exact name."
        return (f"error: no KOSPI or KOSDAQ stock named {stock_name!r}. KRX lists Korean names: use the name exactly "
                "as the user wrote it (e.g. 삼성전자, SK하이닉스). Never guess a ticker: a wrong ticker returns "
                "another company's prices.")
    start, end, days = _window(days)
    try:
        df = stock.get_market_ohlcv(start, end, ticker)
    except Exception as e:  # noqa: BLE001
        return _failed(e)
    if df is None or df.empty:
        return "error: KRX returned no data (KRX may be unreachable, or it is a holiday)"
    return f"{name} ({ticker}, KRX, daily, KRW; the last row is the latest):\n" + _table(
        df, days, {"close": "종가", "high": "고가", "low": "저가", "volume": "거래량"})


if __name__ == "__main__":
    mcp.run(transport="stdio")
"""#
}
