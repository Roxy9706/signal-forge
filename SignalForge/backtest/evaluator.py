
# REAL EVALUATOR
# LONG: TP=any(high>=tp); SL=any(low<=sl)
# SHORT: TP=any(low<=tp); SL=any(high>=sl)
# Same-candle both touched = conservative (documented identical for both directions)
# Expiration: only candles with close in (signal_t, signal_t + horizon]
# Reject: high<low, missing fields, out-of-order, future-to-T
# Gross R; net = gross - fee - slippage
# Signal immutable: input frozen; output separate
