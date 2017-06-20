#!/usr/bin/python
#hist on each cell:
#pd.DataFrame(sim_results.error.iloc[0][0]).hist(bins=10)

#create new data frame with all errors at title on module size. taking only x quants:
q=3
execfile("functions/functions.py")

print "reading csv"
sim_results=pd.read_csv("all.csv").applymap(lambda x: m(x) if type(x)==str and "[" in x and "nan" not in x else x)

print "taking needed data"
df=pd.DataFrame()
sub=sim_results.loc[(sim_results.x_quantizer_number_of_quants==q)]
#for i in range(0,sub.shape[0]): df[sub.iloc[i].x_mod]=(sub.iloc[i].recovered_x_before_alpha-sub.iloc[i].original_y).A1
#for i in range(0,sub.shape[0]): df[sub.iloc[i].x_mod]=(sub.iloc[i].recovered_x).A1
##for i in range(0,sub.shape[0]): df[sub.iloc[i]["cov"].A1[1]]=(sub.iloc[i].recovered_x).A1
for i in range(0,sub.shape[0]): df[sub.iloc[i]["cov"].A1[1]]=(sub.iloc[i].recovered_x_before_alpha-sub.iloc[i].original_y).A1
print "doing hist"
df.hist(bins=180, range=[-3,3])
#df.hist(bins=80)
show()

#or
#for i in range(0,sub.shape[0]): df[sub.iloc[i].x_mod]=sub.iloc[i].error.A1
