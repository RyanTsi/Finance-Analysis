# -*- coding: utf-8 -*-
"""验证新浪价格接口"""
import akshare as ak

try:
    df = ak.stock_hk_daily(symbol="hk00700", start_date="20150101", end_date="20260818", adjust="qfq")
    print("[HK-sina] OK", df.shape, "cols:", list(df.columns))
    print(df.tail(1).to_string())
except Exception as e:
    print("[HK-sina] FAIL:", type(e).__name__, str(e)[:100])

try:
    df = ak.stock_us_daily(symbol="AAPL", adjust="qfq")
    print("[US-sina] OK", df.shape, "cols:", list(df.columns))
    print(df.tail(1).to_string())
except Exception as e:
    print("[US-sina] FAIL:", type(e).__name__, str(e)[:100])
