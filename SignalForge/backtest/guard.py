# CLOSED-CANE GUARD: assert all input candles <= T; future candles rejected explicitly
# Ambiguity: conservative both LONG/SHORT

assert all(c<=T for c in data), 'future candle leak'; if violated: raise ValueError('look-ahead detected')
