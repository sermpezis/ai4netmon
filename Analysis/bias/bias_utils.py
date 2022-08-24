import numpy as np
import pandas as pd
from scipy.stats import kstest
from ai4netmon.Analysis.aggregate_data import data_aggregation_tools as dat

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



def get_distribution_vectors_from_data(target_data, sample_data, data_type, nb_bins=None):
	'''
	Returns two distribution vectors for the given target and sample data (i.e., vector of values that sum to one)
	and a vector with the bins that correspond to each position of the vector.

	If the data (data_type) is numerical, it bins the values in equal <nb_bins> bins. If the data are categorical, 
	the returned bins are the catefories appearing in the data.

	:param 	target_data:	(list / numpy array) values of the "target distribution" (e.g., groundtruth)
	:param 	sample_data:	(list / numpy array) values of the samples, whose bias will be quantified
	:param 	data_type:		(str) type of input data; available options {'numerical', 'categorical'}
									- for 'numerical': params are {'bins', 'alpha'}
									- for 'categorical': params are {'alpha'}
	:return:		(list, list, list)
						- the distribution vector p of the target_data
						- the distribution vector q of the sample_data
						- the bins (for 'numerical') or the categories (for 'categorical') corresponding to 
						  the distribution vectors; the size of bins is nb_bins+1 (for 'numerical') or equal to 
						  the number of categories (for 'categorical')
	'''
	if data_type == 'numerical':
		p, bins = np.histogram(target_data, nb_bins)
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

	return p,q, bins


def get_bias_from_data(method, target_data, sample_data, data_type, **params):
	'''
	Returns the bias metric (KL divergence, Total Variation, or Max variation distance) 
	between two data vectors (target_data and sample_data). 

	:param 	target_data:	(list / numpy array) values of the "target distribution" (e.g., groundtruth)
	:param 	sample_data:	(list / numpy array) values of the samples, whose bias will be quantified
	:param 	data_type:		(str) type of input data; available options {'numerical', 'categorical'}
									- for 'numerical': params are {'bins', 'alpha'}
									- for 'categorical': params are {'alpha'}
	:return:		(float, list, list)
						- the bias metric value
						- the list of elements of the bias metrics (whose sum is the bias metric value 
						  in case of KL divergence and Total Variation, and whose max is the bias metric value
						  in case of Max Variation)
						- the bins (for 'numerical') or the categories (for 'categorical') corresponding to the list of Total Variation values
	'''
	p, q, bins = get_distribution_vectors_from_data(target_data, sample_data, data_type, nb_bins=params.get('bins', None))
	if method == 'kl_divergence':
		bias_score, bias_per_bin = kl_divergence(p, q, alpha=params['alpha'])
	elif method == 'total_variation':
		bias_per_bin = [0.5*np.abs(p[i]-q[i]) for i in range(len(p))]
		bias_score = np.sum(bias_per_bin)
	elif method == 'max_variation':
		bias_per_bin = [np.abs(p[i]-q[i]) for i in range(len(p))]
		bias_score = np.max(bias_per_bin)
	else:
		raise ValueError
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
	if method in ['kl_divergence', 'total_variation', 'max_variation']:
		bias_score, bias_per_bin, bins = get_bias_from_data(method, target_data, sample_data, **params)
	elif method == 'ks_test':
		statistic, p_value = kstest(target_data, sample_data)
		bias_score = 1-p_value
	else:
		raise TypeError

	return bias_score 





def get_feature_type(feature, all_features=False):
	'''
	:param 	feature: 		(str) name of the feature (in case all_features=False) or name of feature category (in case all_features=True)
	:param 	all_features: 	(Boolean) 
	:return:				(str) the type of the given feature (if all_features=False)
							or (list) a list of the features in the given feature category (if all_features=True)
	'''

	FEATURE_TYPES = dict()
	# binary features are included in categorical features
	FEATURE_TYPES['categorical'] =  ['AS_rank_source', 'AS_rank_iso', 'AS_rank_continent', 'is_personal_AS', 'peeringDB_info_ratio', 
	'peeringDB_info_traffic', 'peeringDB_info_scope', 'peeringDB_info_type', 'peeringDB_policy_general', 'is_ris_peer_v4', 
	'is_ris_peer_v6', 'is_routeviews_peer','is_in_peeringDB']
	FEATURE_TYPES['numerical'] =  ['AS_rank_numberAsns', 'AS_rank_numberPrefixes', 'AS_rank_numberAddresses', 'AS_rank_total',
	'AS_rank_customer', 'AS_rank_peer', 'AS_rank_provider', 'peeringDB_ix_count', 'peeringDB_fac_count', 'AS_hegemony', 
	'peeringDB_info_prefixes4', 'peeringDB_info_prefixes6', 'nb_atlas_probes_v4', 'nb_atlas_probes_v6']
	FEATURE_TYPES['ordinal'] = ['AS_rank_rank']
	FEATURE_TYPES['other'] = ['AS_rank_longitude', 'AS_rank_latitude', 'peeringDB_created']

	
	if all_features:
		return FEATURE_TYPES[feature]
	else:
		for ftype, flist in FEATURE_TYPES.items():
			if feature in flist:
				return ftype
		raise ValueError	# if the feature is not found in any type of features



def preprocess_data_series(ds):
	'''
	Receives a pandas.Series and process it, by 
		(i) removing null values
		(ii) if the type of data is numerical, it takes the logarithm of the data and then replaces inf values (due to log(0)) to -0.1
	:param	ds:	(pandas.Series)
	:return:	(pandas.Series)
	'''
	d = ds.copy()
	d = d[(d.notnull())]

	ftype = get_feature_type(d.name)
	if ftype == 'numerical': # pre-processing for the numerical cases
		# d[d<=1] = 0.9#np.nan
		d = np.log(d)
		d[np.isinf(d)] = -0.1
	
	return d



def bias_score_dataframe(target_df, sample_df_dict, preprocess=True, **params):
	bias_df = pd.DataFrame(index=target_df.columns)
	for col in target_df.columns:
		ds1 = target_df[col]
		if preprocess:
			ds1 = preprocess_data_series(ds1)

		for s_name, s_df in sample_df_dict.items():
			ds2 = s_df[col]
			if preprocess:
				ds2 = preprocess_data_series(ds2)
			params['data_type'] = get_feature_type(col)
			bias_df.loc[col, s_name] = bias_score(ds1, ds2, **params)
	return bias_df




def subsampling_naive_sorting(df, set_of_monitors, bias_params, FEATURES):
	network_sets_dict_for_bias = dict()
	for ASN in set_of_monitors:
		network_sets_dict_for_bias[str(ASN)] = df.loc[list(set(set_of_monitors)-set([ASN]))][FEATURES]
	bias_df = bias_score_dataframe(df[FEATURES], network_sets_dict_for_bias, **bias_params)
	bias_diff = bias_df.sum(axis=0)
	return bias_diff.sort_values().index.tolist()

def subsampling_greedy(df, set_of_monitors, bias_params, FEATURES):
	current_set = set(set_of_monitors)
	ordered_list = []
	while len(current_set)>0:
		ASN = subsampling_naive_sorting(df, current_set, bias_params, FEATURES)[0]
		ordered_list.append(ASN)
		current_set.remove(ASN)
	return ordered_list


def subsampling(method, df, set_of_monitors, bias_params, FEATURES):
	'''
	Method to return an ordered list of the monitors in the set_of_monitors in a way that they decrease the bias 
	if they are removed (remove first the first monitors).

	:param 	method:			(str) 'sorting' for naive sorting algorithm or 'greedy'
	:param 	df:				(pandas.DataFrame) the dataframe with the network features; networks are in the index
	:param 	set_of_monitors:(set) set of monitors (i.e. indexes of dataframe), from which we want to subsample
	:param 	bias_params:	(dict) parameters for the bias calculation
	:param 	FEATURES:		(list) list of features (i.e., columns of the dataframe) to be taken into account in the bias calculation

	:return: 	(list) ordered list with monitors; the first monitor in the list is the first monitor to be removed
	'''
	if method == 'sorting':
		return subsampling_naive_sorting(df, set_of_monitors, bias_params, FEATURES)
	elif method == 'greedy':
		return subsampling_greedy(df, set_of_monitors, bias_params, FEATURES)
	else:
		raise TypeError



def get_features_dict_for_visualizations():
	'''
	returns a dict with 
		- keys (str): the feature names in the aggregated dataframe (i.e., column names)
		- values (str): the names to be shown in the visualization plots
	Important: only a subset of features from the aggregated dataframe is returned
	'''
	features_dict = {
	    # Location-related info
	    'AS_rank_source': 'RIR region',
	    'AS_rank_iso': 'Location\n (country)',
	    'AS_rank_continent': 'Location\n (continent)',
	    # network size info
	    'AS_rank_numberAsns': 'Customer cone\n (#ASNs)', 
	    'AS_rank_numberPrefixes': 'Customer cone\n (#prefixes)',
	    'AS_rank_numberAddresses': 'Customer cone\n (#addresses)',
	    'AS_hegemony': 'AS hegemony',
	    # Topology info
	    'AS_rank_total': '#neighbors\n (total)',
	    'AS_rank_peer': '#neighbors\n (peers)', 
	    'AS_rank_customer': '#neighbors\n (customers)', 
	    'AS_rank_provider': '#neighbors\n (providers)',
	    # IXP related
	    # 'is_in_peeringDB': 'In PeeringDB', 
	    'peeringDB_ix_count': '#IXPs\n (PeeringDB)', 
	    'peeringDB_fac_count': '#facilities\n (PeeringDB)', 
	    'peeringDB_policy_general': 'Peering policy\n (PeeringDB)',
	    # Network type
	    'peeringDB_info_type': 'Network type\n (PeeringDB)',
	    'peeringDB_info_ratio': 'Traffic ratio\n (PeeringDB)',
	    'peeringDB_info_traffic': 'Traffic volume\n (PeeringDB)', 
	    'peeringDB_info_scope': 'Scope\n (PeeringDB)',
	    'is_personal_AS': 'Personal ASN', 
	}
	return features_dict



def get_bias_of_monitor_set(df=None, imp='Custom set', monitor_list=None, params=None):
	'''
	Returns a dataframe with the bias values of the given set of monitors (or the given Internet Measurement Platform, IMP).

	:param	df:				(pandas.dataframe) dataframe with the data for all networks (each row); if None is given, loads the latest df. 
	:param	imp:			(str) the name of the Internet Measurement Platform whose bias is needed (valid options 
								for IMPs are ['RIPE RIS', 'RouteViews', 'RIPE Atlas', 'RIPE RIS & RouteViews']); if a different
								value is given, calculate the bias of the set of monitors in monitor_list
	:param	monitor_list: 	(str or list) list of ASNs whose bias is needed (only in case it is different than any of the existing IMPs)
	:param 	params:			(dict) parameters for the bias calculation; 
								default is {'method':'kl_divergence', 'bins':10, 'alpha':0.01}

	:return:	(pandas.dataframe) a dataframe with index the bias dimensions, and with 1 column with the given set of monitors. 
					Values are the bias scores along each dimension.
	'''

	if df is None:
		df = dat.load_aggregated_dataframe(preprocess=True)

	if imp == 'RIPE RIS':
		df_mon = df.loc[(df['is_ris_peer_v4']>0) | (df['is_ris_peer_v6']>0), :]
	elif imp == 'RouteViews':
		df_mon = df.loc[df['is_routeviews_peer']>0, :]
	elif imp == 'RIPE Atlas':
		df_mon = df.loc[(df['nb_atlas_probes_v4']>0) | (df['nb_atlas_probes_v6']>0), :]
	elif imp == 'RIPE RIS & RouteViews':
		df_mon = df.loc[(df['is_ris_peer_v4']>0) | (df['is_ris_peer_v6']>0) | (df['is_routeviews_peer']>0), :]
	else:
		if monitor_list is None:
			raise ValueError('Either a valid IMP name is needed or a non-empty monitor_list')
		else:
			df_mon = df.loc[[i for i in monitor_list if i in df.index], :]

	
	FEATURE_NAMES_DICT = get_features_dict_for_visualizations()
	FEATURES = list(FEATURE_NAMES_DICT.keys())

	if params is None:
		params = {'method':'kl_divergence', 'bins':10, 'alpha':0.01}

	bias_df = bias_score_dataframe(df[FEATURES], {imp: df_mon[FEATURES]}, **params)

	return bias_df


def get_bias_of_all_imps(df=None, params=None):
	'''
	Returns a dataframe with the bias values for all IMPs
	:param	df:				(pandas.dataframe) dataframe with the data for all networks (each row) 
	:param 	params:			(dict) parameters for the bias calculation; 
								default is {'method':'kl_divergence', 'bins':10, 'alpha':0.01}

	:return:	(pandas.dataframe) a dataframe with index the bias dimensions, and with columns the different IMPs
	'''

	DEFAULT_IMP_OPTIONS = ['RIPE RIS', 'RouteViews', 'RIPE Atlas', 'RIPE RIS & RouteViews']
	bias_df = pd.DataFrame()
	for imp in DEFAULT_IMP_OPTIONS:
		bias_df = pd.concat([bias_df, get_bias_of_monitor_set(df=df, imp=imp, params=params)], axis=1)

	return bias_df