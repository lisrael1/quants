import pandas as pd
import glob
from tqdm import tqdm
df=pd.concat([pd.read_csv(i,index_col=[0]) for i in tqdm(glob.glob("res*.csv.gz"))])
print('done reading csv files. will take about 1 minute extra to save this file')
# df.to_csv('final_results.csv.gz',compression='gzip') too slow
df.to_csv('final_results.csv')
print('done. now you can run pigz on it and send it to ~/www/ -> \npigz final_results.csv;mv final_results.csv.gz ~/www/ \nthen download it from http://www.cs.huji.ac.il/~lisrael1/final_results.csv.gz')