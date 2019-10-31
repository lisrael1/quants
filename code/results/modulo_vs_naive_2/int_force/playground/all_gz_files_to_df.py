# run with python3 -i
# then do what you want with df
import pandas as pd
import numpy as np
pd.set_option("display.max_columns",1000)
pd.set_option('expand_frame_repr',False)
pd.set_option('display.max_colwidth',1000)
from tqdm import tqdm
from glob import glob
import pickle
import _pickle
import timeit


alls=[]
for f in tqdm(glob(r"*gz")):
	try:
		tmp=pd.read_csv(f,index_col=[0])#, nrows=200000)
		alls+=[tmp]
	except:
		print('failed to read '+f)
print('now doint concatenation')
df=pd.concat(alls, ignore_index=True, sort=False)
#df=pd.DataFrame(np.vstack([a.values for a in alls]))

if 0:
        print('write csv time')
        print(timeit.timeit("df.to_csv('del.csv')", number=1, setup="from __main__ import df"))
if 0:
        print('write pickle time')
        f = open('del.pckl', 'wb')
        print(timeit.timeit("pickle.dump(df, f)", number=1, setup="from __main__ import df,f,pickle"))
        f.close()


# print('read pickle')
# f = open('del.pckl','rb')
# df = pickle.load(f)
# f.close()

