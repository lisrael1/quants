#!/usr/bin/python
execfile("functions/functions.py")

#next TODO:
#try to do SNR minus quantization SNR
#probably you better put a quant at the 0, but then you will always will have odd number of quants


"""
we have here only 2 inputs, y and x, and each input has its own modulo size and quantization
we want to see how the change of y modulo and quantization replects on x mse

gvim shortcuts:
zf create folding
za toggle folding state

"""
best_bin_sizes_at_1_var={1: 0, 2: 0.0616666, 3: 1.64, 4: 1.342, 5: 1.186666, 6: 1.0021428571428579, 7: 0.91375, 8: 0.838888, 9: 0.7565, 10: 0.660454, 11: 0.6333333, 12: 0.57346153846153891, 13: 0.535, 14: 0.5106666, 15: 0.4821875, 16: 0.4529411764705884, 17: 0.434166666, 18: 0.415263157894, 19: 0.38925, 20: 0.36142857142857165, 21: 0.368181818181, 22: 0.34456521739130475, 23: 0.32666666, 24: 0.3184, 25: 0.30692307692307708, 26: 0.28555555555, 27: 0.28392857142857153, 28: 0.27741379310344838, 29: 0.265833333, 30: 0.25983870967741973, 31: 0.245625, 32: 0.2360606060606063, 33: 0.23573529411764715, 34: 0.21785714285714297, 35: 0.22208333333333341, 36: 0.21310810810810832, 37: 0.21381578947368451, 38: 0.20243589743589752, 39: 0.188}

max_x_bin_number=20
##max_x_bin_number=10#for fast estimation
min_x_bin_number=2
##min_x_bin_number=5#for fast estimation

number_of_samples=4e6#slow, only for server
number_of_samples=4e4#slow
number_of_samples=2.8e7#supper slow
number_of_samples=4e1#fast estimation
##number_of_samples=1#put here 1 and look at all_data.csv to see the flow on single sample

modulo_jumps_resolution=0.05
#modulo_jumps_resolution=0.5#for fast estimation
min_modulo_size=5.5
##min_modulo_size=7#from 5 quants, the modulo starts at 7

cov=mat([[10,9.5],
         [9.5,10]])

#preparing the data - try to fix this to search for the best one:
def prepare_sim_args():
    print "simulation time start creating quantizers: ",time() - start_time,"sec"
    def x_args():
        if 0:#take known best quantizer instead of looking for them
        	qx=[simple_quantizer(number_of_quants=i,bin_size=cov[0,0]*best_bin_sizes_at_1_var[i]) for i in range(min_x_bin_number,max_x_bin_number)]
        if 1:#trying all modulo sizes (at relevant range of 4 to 9):
        	qx=[simple_quantizer(number_of_quants=j,bin_size=i/j) for i in arange(min_modulo_size,9,modulo_jumps_resolution) for j in range(min_x_bin_number,max_x_bin_number)]
        if 0:#try just a few numbers to see how the flow runs
        	qx=[quantizer(number_of_quants=5,bin_size=6.6/5) for i in range(20)] #put number_of_samples=1 and independed_var=100
        return qx

    def y_args():
    	#now lets pick y quantizer:
    	if 0:
    		qy=[simple_quantizer(number_of_quants=2000,bin_size=i) for i in [0.5,1,1.5,2,2.5,3,4,5]]
    	else:#perfect Y
    		qy=[simple_quantizer(number_of_quants=200000,bin_size=1)]
        return qy

    sim_args=[sim_2_inputs(
		number_of_samples=number_of_samples,#dont put above 4e5
		cov=cov,
		x_quantizer=i,
		y_quantizer=j,
		dither_on=0
		) for i in x_args() for j in y_args()]
    print "simulation time2: ",time() - start_time,"sec"
    return sim_args

def run_sim(sim_args):
    list_of_df=matching(run_single_sim,sim_args)
    sim_results=pd.concat(list_of_df,axis=0)
    return sim_results

#running on best mse for each number of quants:
def parse_sim_results(sim_results):
	print "simulation time3: ",time() - start_time,"sec"

	if 1:#basic one, exp plot
		plot_threads="y_quantizer_bin_size"
		x_plot='x_quantizer_number_of_quants'
		y_sort="normalized_mse"
		y_plot=y_sort
##  		y_plot="x_quantizer_modulo_edge_to_edge"#to see the modulo size
	if 0:#spliting by x_quantizer_number_of_quants
		plot_threads="x_quantizer_number_of_quants"
		x_plot='x_quantizer_modulo_edge_to_edge'
		y_sort="normalized_mse"
		y_plot=y_sort#doesnt matter because we dont have duplications at x

##	sim_results_table=pd.DataFrame([i.dict() for i in sim_results])
	sim_results_table=sim_results.sort(columns=[x_plot,y_sort]).reset_index().drop('index',1)#sorting from A to Z
	print "data ready,",sim_results_table.index.size,"lines"
##	return sim_results_table#temp debug
	if sim_results_table.index.size<100:
		sim_results_table.transpose().to_csv("temp/all_data.csv")#we will see each sample at different column
	else:
		sim_results_table.to_csv("temp/all_data.csv")
	thread_options=set(sim_results_table[plot_threads].tolist())
	for i in thread_options:
		thread_in_sim_results_table=sim_results_table.loc[sim_results_table[plot_threads]==i]
		thread_in_sim_results_table=thread_in_sim_results_table.sort(columns=[x_plot,y_sort])#sorting from A to Z
		thread_in_sim_results_table=thread_in_sim_results_table.drop_duplicates(subset=x_plot,take_last=False)#take the first one, lowest mse
		plot(thread_in_sim_results_table[x_plot],thread_in_sim_results_table[y_plot],label=i)
		if len(thread_options)==1:
			text(15,0.,thread_in_sim_results_table[[x_plot,y_plot]].values)
##		print thread_in_sim_results_table[[x_plot,y_plot]]
		if 0:#for ploting the plot threads in different plots
			xlabel(x_plot)
			ylabel(y_plot)
			title("number of quants ="+str(i))
			grid()
			savefig("temp"+dlmtr+"mse per modulo"+dlmtr+"mse vs mod size at "+str(i)+" bins"+img_type)
			close()
	xlabel(x_plot)
	ylabel(y_plot)
	title("-")
	if len(thread_options)>1:
		legend(loc="best", shadow=True, title=plot_threads)
	grid()
	print "simulation time before show: ",time() - start_time,"sec"
	#show()
	savefig("del.jpg")
	return sim_results_table

if 1:
    #run sim
    sim_results=run_sim(prepare_sim_args())
    sim_results.to_csv('temp/sim_results.csv')
if 1:
	#parse sim results:
	sim_results=pd.read_csv('temp/sim_results.csv')
	sim_results_table=parse_sim_results(sim_results)
print "simulation time: ",time() - start_time,"sec"


