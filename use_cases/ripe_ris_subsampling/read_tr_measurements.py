import json
from pprint import pprint
from collections import defaultdict
import numpy as np 
from matplotlib import pyplot as plt
from matplotlib import image as img
import random
from PIL import Image


def get_image_data():
	im = np.array(Image.open('image.png'))
	im = im[7:279, 45:712,:]

	hh = im.shape[0]
	ww = im.shape[1]
	Col = 255*np.array([0.12156863,0.46666667,0.70588235,1.])
	bnd = 1

	xmax_lin = np.log10(1)
	xmin_lin = np.log10(500)
	xax = np.linspace(xmax_lin,xmin_lin,ww)
	e = 0.05
	yax = np.linspace(1+e,0-e,hh)

	for i in range(hh):
	    for j in range(ww):
	        im[i,j,:] = 255*(any(im[i,j,:]>Col+bnd) or any(im[i,j,:]<Col-bnd))

	mapim = np.abs(im[:,:,0]/255-1).astype(bool)

	yval = np.array([np.average(yax[mapim[:,t]]) for t in range(ww)])
	xax = 10**xax

	for i,v in enumerate(list(yval)):
		if (i<50) and (v > yval[50]):
			yval[i] = np.nan

	# plt.plot(xax, yval)
	# plt.xscale('log')
	# e = 0.025
	# plt.axis([1,500,0-e,1+e])
	# plt.grid(True)
	# plt.show()
	return xax, yval



MSMT_FNAME = 'RIPE-Atlas-measurement-40508755.json'
with open(MSMT_FNAME, 'r') as f:
	data = json.load(f)
# print(len(data))
# pprint(data[0])


ATLAS_INFO_FNAME = '../../data/misc/RIPE_Atlas_probes_info.json'
with open(ATLAS_INFO_FNAME, 'r') as f:
	atlas_info = json.load(f)
# print(len(atlas_info))
# pprint(atlas_info[0])
prb2asn = dict()
for prb in atlas_info:
	if prb.get('asn_v4') is not None:
		prb2asn[prb['id']] = prb['asn_v4']
print(len(prb2asn))

SET_LISTS_FNAME = 'Lists_of_Atlas_samples.json'
# with open(SET_LISTS_FNAME, 'r') as f:
# 	sets = json.load(f)

# with open('./data/sorted_list_greedy_rnd300_Atlas.json', 'r') as f:
# 	greedy_sel = json.load(f)
# sets['Atlas rnd 300'] = [int(float(i)) for i in greedy_sel]
# sets['Atlas sub 200'] = [int(float(i)) for i in greedy_sel[-200:]]
# sets['Atlas rnd 101'] = random.sample([int(float(i)) for i in greedy_sel],100)


with open('./data/sorted_list_greedy_Atlas.json', 'r') as f:
	greedy_sel = json.load(f)
sets = dict()
sets['Atlas ALL'] = [int(float(i)) for i in greedy_sel]
# sets['Atlas sub 1000'] = [int(float(i)) for i in greedy_sel[-1000:]]
# sets['Atlas sub 300'] = [int(float(i)) for i in greedy_sel[-300:]]
# sets['Atlas sub 100'] = [int(float(i)) for i in greedy_sel[-100:]]

# for i in range(10):
# 	sets['Atlas rnd {}'.format(300+i)] = random.sample(sets['Atlas ALL'],300)
# 	# sets['Atlas rnd 100'] = random.sample(sets['Atlas ALL'],100)

LIST_SAMPLE = [100*i for i in list(range(1,30))]+[len(greedy_sel)]
for i in LIST_SAMPLE:
	sets['Atlas sub {}'.format(i)] = [int(float(j)) for j in greedy_sel[-i:]]
	sets['Atlas rnd {}'.format(i)] = [int(float(j)) for j in random.sample(greedy_sel,i)]


prb2ttl = dict()
asn2ttl = defaultdict(list)
for ms in data:
	prb_id = ms['prb_id']
	ttl = ms['avg']
	prb2ttl[prb_id] = ttl
	if prb_id in prb2asn:
		asn2ttl[prb2asn[prb_id]].append(ttl) 


bins=np.logspace(0,2.5,15)
# bins=np.linspace(0,400,100)
hist = dict()
for s,l in sets.items():
	pdf_med = []
	for asn in l:
		pdf_med.append(np.median(asn2ttl[asn]))
	hist[s],_ = np.histogram(pdf_med, bins=bins)
	hist[s] = hist[s]/sum(hist[s])

xcdn, ycdn = get_image_data()
xcdn = [x for i,x in enumerate(xcdn) if ~np.isnan(ycdn[i])]
ycdn = [y for i,y in enumerate(ycdn) if ~np.isnan(ycdn[i])]

hist['CDN'] = []
Ym = 0
for i in range(1,len(bins)):
	bM = bins[i]
	for j,v in enumerate(xcdn):
		if v>=bM:
			break
	hist['CDN'].append(np.nanmax([ycdn[j-1]-Ym,0]))
	Ym = ycdn[j-1]
# print(sum(hist['CDN']))
# print(bins)
# pprint(hist)


from ai4netmon.Analysis.bias import bias_utils as bu
tv = lambda p,q: np.sum([0.5*np.abs(p[i]-q[i]) for i in range(len(p))])
bias = {k:bu.kl_divergence(hist['CDN'],q,alpha=0.01)[0] for k,q in hist.items()}
# bias = {k:tv(hist['CDN'],q) for k,q in hist.items()}
# pprint(bias)
# exit()


bf = lambda x,y: bu.kl_divergence(x,y,alpha=0.01)[0]
# bf = lambda x,y: tv(x,y)
# bf = lambda x,y: np.sum([0.5*(bins[i-1]+bins[i])*(x[i-1]-y[i-1]) for i in range(1,len(bins))])
BBIAS = []
for i in LIST_SAMPLE:
	BBIAS.append(bf(hist['CDN'],hist['Atlas sub {}'.format(i)]))
plt.plot(LIST_SAMPLE, BBIAS, label='selected subsets')
BBIAS = []
for i in LIST_SAMPLE:
	BBIAS.append(bf(hist['CDN'],hist['Atlas rnd {}'.format(i)]))
plt.plot(LIST_SAMPLE, BBIAS, label='random subsets')
plt.xscale('log')
plt.xlabel('#probes')
plt.ylabel('difference from CDN data')
plt.title('Case A: difference in entire distribution')
# plt.title('Case B: difference in mean value')
plt.show()
plt.savefig('fig_diff_CDN_KL.png')
# plt.savefig('fig_diff_CDN_AVG.png')



colors = ['r','b','g','m','c','y']
i = 0
plot_sets = {s:sets[s] for s in ['Atlas sub 1000', 'Atlas sub 300', 'Atlas rnd 1000', 'Atlas rnd 300', 'Atlas ALL'] }
for s,l in plot_sets.items():
	j = 0
	pdf_all = []
	pdf_med = []
	pdf_avg = []
	pdf_rnd = []
	pdf_min = []
	for asn in l:
		pdf_all.extend(asn2ttl[asn])
		pdf_med.append(np.median(asn2ttl[asn]))
		pdf_avg.append(np.mean(asn2ttl[asn]))
		if len(asn2ttl[asn]) >0 :
			pdf_rnd.extend(random.sample(asn2ttl[asn],1))
			pdf_min.append(np.min(asn2ttl[asn]))
		else:
			j += 1
	plt.plot()
	plt.hist(pdf_med, bins=1000, density=True, cumulative=True, label=s,
         histtype='step', alpha=0.8, color=colors[i])
	# plt.hist(pdf_rnd, bins=1000, density=True, cumulative=True, label=s,
 #         histtype='step', alpha=0.8, linestyle='--', color=colors[i])
	# plt.hist(pdf_all, bins=1000, density=True, cumulative=True, label=s,
 #         histtype='step', alpha=0.8, linestyle='--', color=colors[i])
	# plt.hist(pdf_avg, bins=1000, density=True, cumulative=True, label=s,
 #         histtype='step', alpha=0.8, linestyle='--', color=colors[i])
	# plt.hist(pdf_min, bins=1000, density=True, cumulative=True, label=s,
 #         histtype='step', alpha=0.8, linestyle='--', color=colors[i])
	i += 1
	print('{}: no samples {}'.format(s,j))


plt.plot(xcdn,ycdn,'k',label='CDN')
e = 0.025
plt.axis([1,500,0-e,1+e])
plt.xscale('log')
plt.xlabel('Median RTT (ms)')
plt.ylabel('CDF over ASNs')
# plt.axis([0,200,0-e,1+e])
plt.legend()
plt.grid(True)
plt.show()
# plt.savefig('fig_TEST_{}.png'.format('ALL'))
plt.close()
	