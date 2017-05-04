#!/usr/bin/python
#run like this:
#./parse_sim_2_inputs.py temp/sim_results_try.csv

'''
hist on each cell:
pd.DataFrame(sim_results.error.iloc[0][0]).hist(bins=10)

create new data frame with all errors at title on module size. taking only 15 quants:

execfile("functions/functions.py")
sim_results=pd.read_csv("/tmp/all.csv").applymap(lambda x: m(x) if type(x)==str and "[" in x and "nan" not in x else x)
df=pd.DataFrame()
sub=sim_results.loc[(sim_results.x_quantizer_number_of_quants==3)]
for i in range(0,sub.shape[0]): df[sub.iloc[i].x_mod]=(sub.iloc[i].recovered_x_before_alpha-sub.iloc[i].original_y).A1

df.hist(bins=100)
show()

or	
for i in range(0,sub.shape[0]): df[sub.iloc[i].x_mod]=sub.iloc[i].error.A1
'''

execfile("functions/functions.py")

#running on best mse for each number of quants:
def parse_sim_results(sim_results):
	print "simulation time3: ",time() - start_time,"sec"

	if 1:#basic one, exp plot
		plot_threads="y_quantizer_bin_size"
		x_plot='x_quantizer_number_of_quants'
		y_sort="normalized_mse"
		y_plot=y_sort
        if 0:#1 for plotting the modulo size and o for plotting the mse
  		    y_plot="x_quantizer_modulo_edge_to_edge"#to see the modulo size
	if 0:#multi plots by x_quantizer_number_of_quants
		plot_threads="x_quantizer_number_of_quants"
		x_plot='x_quantizer_modulo_edge_to_edge'
		y_sort="normalized_mse"
		y_plot=y_sort#doesnt matter because we dont have duplications at x
	if 0:#plot the mse for each modulo size (to see how accurate the sim choose the modulo size) you better run the sim for this with modulo size from 0 to 16 with jumps of 0.05
            plot_threads="x_quantizer_number_of_quants"
            if 1:
                sim_results=sim_results.loc[sim_results.x_quantizer_number_of_quants==7]#viewing only x number of quants
            x_plot='x_quantizer_modulo_edge_to_edge'
            y_sort="normalized_mse"
            y_plot=y_sort


	sim_results_table=sim_results.sort(columns=[x_plot,y_sort]).reset_index().drop('index',1)#sorting from A to Z
	print "data ready,",sim_results_table.index.size,"lines"
##	return sim_results_table#temp debug
	if sim_results_table.index.size<100:
		sim_results_table.transpose().to_csv("temp/temp_all_data.csv")#we will see each sample at different column
	else:
		sim_results_table.to_csv("temp/temp_all_data.csv")
	thread_options=sorted((set(sim_results_table[plot_threads].tolist())))
	for i in thread_options:
		thread_in_sim_results_table=sim_results_table.loc[sim_results_table[plot_threads]==i]
		thread_in_sim_results_table=thread_in_sim_results_table.sort(columns=[x_plot,y_sort])#sorting from A to Z
		thread_in_sim_results_table=thread_in_sim_results_table.drop_duplicates(subset=x_plot,take_last=False)#take the first one, lowest mse
		plot(thread_in_sim_results_table[x_plot],thread_in_sim_results_table[y_plot],label=i)
		if len(thread_options)==1:
			print thread_in_sim_results_table[y_plot].tolist()
			text(mean(thread_in_sim_results_table[x_plot]),min(thread_in_sim_results_table[y_plot]),thread_in_sim_results_table[[x_plot,y_plot]].values)
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
	if 0:
		import matplotlib
		matplotlib.use('Agg')
		savefig("temp/del.jpg")
	else:
		show()
	return sim_results_table

#parse sim results:
sim_results=pd.read_csv(argv[1]).applymap(lambda x: m(x) if type(x)==str and "[" in x and "nan" not in x else x)
sim_results_table=parse_sim_results(sim_results)
print "simulation time: ",time() - start_time,"sec"


