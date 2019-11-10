# run with python3 -i ../all_gz_files_to_df.py
# or by sbatch --mem=75000m --time=23:50:0 --wrap 'python3 ../all_gz_files_to_df.py;date|mutt -s done israelilior@gmail.com'
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
import os

class timer:
    def __init__(self):
        import time
        self.time=time
        self.start = time.time()
    def current(self):
        return "{:} sec".format(self.time.time() - self.start)
    def __del__(self):
        print("closing timer with value {:}".format(self.current()))
        return self.current()  
    def __exit__(self, exception_type, exception_value, traceback): pass
    def __enter__(self): pass


alls=[]
for f in tqdm(glob(r"*gz")):
    try:
        tmp=pd.read_csv(f,index_col=[0])#, nrows=200000)
        alls+=[tmp]
    except:
        print('failed to read '+f)
with timer():
    print('now doint concatenation')
    df=pd.concat(alls, ignore_index=True, sort=False)
    #df=pd.DataFrame(np.vstack([a.values for a in alls]))

        
        
with timer():
    pvt=df.groupby(['method','quant_size','snr','number_of_bins']).rmse.mean().reset_index()
    pvt.to_csv('~/www/pivot_rmse_%s.csv'%pd.datetime.now().strftime('%d_%m_%Y_%H.%M.%S'))

        
        
print('write pickle')
with timer():
    print('saving pkl file')
    os.makedirs('/tmp/lisrael1_del/',exist_ok=True)
    df.to_pickle('/tmp/lisrael1_del/del.pkl')
with timer():
    print('pigz on pkl file')
    print(os.popen('pigz /tmp/lisrael1_del/del.pkl').read())
print(os.popen('mv /tmp/lisrael1_del/del.pkl.gz ~/www/').read())


# print('read pickle')
# f = open('del.pckl','rb')
# df = pickle.load(f)
# f.close()


