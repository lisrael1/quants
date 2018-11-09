'''
you can use instead:
    pigz -dck *.gz $q/results/modulo_vs_naive_2/*.gz|grep -v quant|pigz>/tmp/lior_tmp.csv.gz
    cat ./results_del_00000000.csv.gz /tmp/lior_tmp.csv.gz >/tmp/lior_tmp2.csv.gz
    split -n l/2 -d /tmp/lior_tmp2.csv.gz /tmp/lior_
    cp /tmp/lior_00 ~/www/final_results.csv.gz
    #download...
    cp /tmp/lior_01 ~/www/final_results.csv.gz
    #download...
'''
import pandas as pd
import glob
from tqdm import tqdm
from optparse import OptionParser

parser = OptionParser(version="%prog 1.0 beta")
parser.add_option("-i", dest="input_directories", type="str", default='./', help='input directories where you have the csv.gz files. you can put many with , between like ../a,b [default: %default]')
parser.add_option("-o", dest="output_folder", type="str", default='./', help='output folder, where the final_results.csv will be. dont forget the / at the end[default: %default]')
(u, args) = parser.parse_args()

files=[]
for folder in u.input_directories.split(','):
    ff=glob.glob(folder+"/res*.csv.gz")
    print('%d files in folder %s'%(len(ff),folder))
    files+=ff
print('now reading files')
df=pd.DataFrame()
for csv in tqdm(files):
        try:
                df=pd.concat([df,pd.read_csv(csv,index_col=[0])])
        except:
                print('skipping file %s'%csv)
print('done reading csv files. will take about 1 minute extra to save this file')
# df.to_csv('final_results.csv.gz',compression='gzip') too slow
df.to_csv(u.output_folder+'final_results.csv')
print('done. now you can run pigz on it and send it to ~/www/ -> \npigz %s/final_results.csv;mv %s/final_results.csv.gz ~/www/ \nthen download it from http://www.cs.huji.ac.il/~lisrael1/final_results.csv.gz'%(u.output_folder,u.output_folder))