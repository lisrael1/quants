#!/usr/bin/python


"""
system goes like this:
basic system:
	encoder:
		modulo input x
			modulo size by var(x-y)
	decoder 1:
		multiply [x,y] in A 
		modulo
	decoder 2:
		multiply at the inverse of A
full system:
	input generator:
		by cov matrix
	encoder 
		multipy by alpha
		add dither
		modulo
		quantisize
	the decoder has 2 blocks:
	decoder 1:
		subtitude by dither 
		subtitude by alpha y 
			or any other integer combinations - multiply by A (here its [1,-1]) so it will be inside the modulo
			to get linear algebra we probably better use A=[[1,-1],[-1,1]]
		modulo
		multiply by alpha

		here we actually get mod(x-y) and we hope that it's inside the modulo
	decoder 2:
		add y (the inverse of A)


alpha is var(x-y) / [var(x-y)+var(dither)] - wielner coefficient

opens:
	modulo depends on the statistice - on the cov so it might be that we dont have the same statistics but we want a known modulo size ahead so we want same modulo size to all
	so all inputs have the same modulo size, number of bins
	but what about var and alpha?
	
"""
class data:
	original_data=[]
	incriptor_output=[]
	dither=0
	mod_size=0
	mse=0
	var=0
	number_of_bins=0
	alpha=0
	dither_on=True
	modulo_on=True

