import numpy as np
import pandas as pd
from scipy.stats import kstest

def kl_divergence(p,q,alpha=0):
	'''
	Returns the KL distance between the two distributions p1 and p2. In case the parameter "alpha" is >0, 
	it returns a smoothed version of the KL distance [Steck18]. 
	If the KL distance diverges (becomes inf; only when a=0) then the function returns the value np.nan

	[Steck18] Harald Steck. 2018. Calibrated recommendations. In ACM RecSys. ACM, 154–162

	:param 	p:		(list / numpy vector) values of the "target" distribution (e.g., groundtruth)
	:param 	q:		(list / numpy vector) values of the "approximation" distribution (e.g., to be checked for bias) - same size with p !!
	:param 	alpha: 	(float) parameter for normalization; a typical value is a=0.01 [Steck18]
	:return:		(float, list) 
						- the KL divergence value (if a>0, then the returned value is in [0,1])
						- the list of elements of KL divergence (whose sum is the KL divergence value)
	'''
	d = [np.nan]*len(p) # initialize the list with the kl distance per item in the distribution
	q_smooth = [(1-alpha)*q[i]+ alpha*p[i]  for i in range(len(p))]
	for i in range(len(p)):
		if p[i] == 0:
			d[i] = 0
		else:
			if q_smooth[i] == 0:
				d[i] = np.nan # the KL metrics diverges (becomes inf)
			else:
				d[i] = p[i]*np.log(p[i]/q_smooth[i])

	if alpha > 0:
		d = d/np.log(1/alpha)

	kl_div = np.sum(d)
		
	return kl_div, d 


def kl_divergence_from_data(target_data, sample_data, data_type, **params):
	'''
	Returns the KL distance between two data vectors (target_data and sample_data). 
	If the data (data_type) is numerical, it calculates the   distributions p1 and p2.


	:param 	target_data:	(list / numpy array) values of the "target distribution" (e.g., groundtruth)
	:param 	sample_data:	(list / numpy array) values of the samples, whose bias will be quantified
	:param 	data_type:		(str) type of input data; available options {'numerical', 'categorical'}
	:param 	params:			(dict) dictionary with params for the selected method
									- for 'numerical': params are {'bins', 'alpha'}
									- for 'categorical': params are {'alpha'}
	:return:		(float, list, list)
						- the KL divergence value
						- the list of elements of KL divergence (whose sum is the KL divergence value)
						- the bins (for 'numerical') or the categories (for 'categorical') corresponding to the list of KL divergence values
	'''
	if data_type == 'numerical':
		p, bins = np.histogram(target_data, params['bins'])
		p = p / len(target_data)
		q, bins = np.histogram(sample_data, bins)
		q = q / len(sample_data)
	elif data_type == 'categorical':
		p = pd.DataFrame(target_data).value_counts()
		bins = list(p.index)
		p = p / len(target_data)
		p = p.values
		q = pd.DataFrame(sample_data).value_counts()
		q = q.reindex(index=bins, fill_value=0)	# use the rows of q in the order of rows in p; for non-existing indices in q, it sets 0  
		q = q / len(sample_data)
		q = q.values
	else:
		raise ValueError
	bias_score, bias_per_bin = kl_divergence(p, q, alpha=params['alpha'])
	return bias_score, bias_per_bin, bins



def bias_score(target_data, sample_data, method='kl_divergence', **params):
	'''
	Returns a value that quanitfies the bias of the distribution of the values in the vector "sample_data" wrt the distribution
	of the values in the vector "target_data", using either
		- the Kullback-Leibler divergence ('kl_divergence')
		- the 1-p_value of a Kolmogorov-Smirnoff test ('ks_test')
	:param 	target_data:	(list / numpy array) values of the "target distribution" (e.g., groundtruth)
	:param 	sample_data:	(list / numpy array) values of the samples, whose bias will be quantified
	:param 	method:			(str) method to calculate bias score; available options {'kl_divergence', 'ks_test'}
	:param 	params:			(dict) dictionary with params for the selected method
									-- for 'kl_divergence': params are {'data_type', 'bins', 'alpha'}
	:return:		(float, list)  the KL divergence value, and the list of elements of KL divergence
	'''
	if method == 'kl_divergence':
		bias_score, bias_per_bin, bins = kl_divergence_from_data(target_data, sample_data, **params)
	elif method == 'ks_test':
		statistic, p_value = kstest(target_data, sample_data)
		bias_score = 1-p_value
	else:
		raise TypeError

	return bias_score 