import pandas as pd
df=pd.concat([pd.read_csv('resutls_%05d.csv'%i,index_col=[0]) for i in range(100)])
print('done reading csv files')
df.to_csv('final_results.csv')