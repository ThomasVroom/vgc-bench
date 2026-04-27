import sys
import pandas as pd
import numpy as np

SHAPE = (10, 10) # shape of results

# 1. put results in rega.txt, regb.txt, etc.
# 2. call this script as `python convert_results.py rega`, etc.

with open(f"./results/{sys.argv[1]}.txt", 'r') as f:
    s = f.read()
    s = s.replace('[', '').replace(']','')
    a = np.fromstring(s, dtype=np.float32, sep=' ').reshape(SHAPE)
    pd.DataFrame(a).to_clipboard(index=False, header=False) # copies to clipboard
