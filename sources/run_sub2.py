from selectlearner import corrcoefs
import numpy as np
from drugconcentrations import get_drugs
from load import read_sub2
from pylab import plot
data, drugs, times, concentrations, gene_names = read_sub2()
first_line = 'Drug Combination,Rank'


names = np.array([n for _,n in gene_names])
positions = names.argsort()
data = data[positions]
names = names[positions]
active = 0
c = 1
selected = np.ones(len(data), bool)
prev = names[-1]
for i,n in enumerate(names):
    if n != prev:
        if c > 1:
            data[active] /= float(c)
        active = i
        c = 1
    else:
        data[active] += data[i]
        selected[i] = 0
        c += 1
    prev = n

data = data[selected]
names = names[selected]


def threshold(data):
    t = 1.5
    dmso = data[:,drugs == 'DMSO']
    media = data[:,drugs == 'Media']
    thresh_up_media = media.mean(1) + t*media.std(1)
    thresh_down_media = media.mean(1) - t*media.std(1)
    thresh_up_dmso = dmso.mean(1)+ t*dmso.std(1)
    thresh_down_dmso = dmso.mean(1)- t*dmso.std(1)

    thresh_up = np.maximum(thresh_up_dmso, thresh_up_media)
    thresh_down = np.minimum(thresh_down_dmso, thresh_down_media)
    return thresh_up, thresh_down_dmso

thresh_up, thresh_down = threshold(data)

drugdata = get_drugs()
valid_data = []

sorted_drugs = sorted(set(drugs))

for dr in sorted_drugs:
    ddata = data[:,drugs ==dr]
    dtimes = times[:,drugs == dr]
    valid = (ddata > thresh_up[:,None]) | (ddata < thresh_down[:,None])
    validinall = np.zeros(len(valid), bool)
    validinall |= valid[:].mean(1) > .5
    valid_data.append(validinall)
    print '{0:32}{1: 8}'.format(dr, np.sum(validinall))
   
def retrieve_gos(names):
    import waldo
    import waldo.uniprot
    from waldo.go import vocabulary
    vocccache = {}
    def cvocabulary(g):
        try:
            return vocccache[g]
        except KeyError:
            v = vocabulary(g)
            vocccache[g] = v
            return v
    gos = []
    for name in names:
        e = waldo.uniprot.retrieve.retrieve_entry(name+'_HUMAN')
        if e is None:
            gos.append([])
        else:
            anns = [ann.go_id for ann in e.go_annotations]
            anns = [a for a in anns if cvocabulary(a) in ['molecular_function']]
            gos.append(anns)
    return gos


def build_gos_pergos(data, names):
    gos = retrieve_gos(names)
    allgos = set()
    for g in gos:
        allgos.update(g)
    allgos = list(allgos)
    perg = np.zeros((len(allgos), data.shape[1]))
    for gs,d in zip(gos,data):
        for g in gs:
            gi = allgos.index(g)
            perg[gi] += d
    return allgos, perg

allgos, perg = build_gos_pergos(data,names)

thresh_up, thresh_down = threshold(perg)
drugdata = get_drugs()
valid_perg = []
for dr in sorted_drugs:
    ddata = perg[:,drugs==dr]
    dtimes = times[:,drugs == dr]
    valid = (ddata > thresh_up[:,None]) | (ddata < thresh_down[:,None])
    validinall = np.zeros(len(valid), bool)
    validinall |= valid.mean(1) > .5
    valid_perg.append(validinall)
    print '{0:32}{1: 8}'.format(dr, np.sum(validinall))


GeneCs = np.array([corrcoefs(valid_data, v) for v in valid_data])
GoCs = np.array([corrcoefs(valid_perg, p) for p in valid_perg])



X = GoCs.ravel()
y = GeneCs.ravel()
y = y[X.ravel() <= .99]
X = X[X <= .99]
X = np.array([X,np.ones(len(X))])
x,r,_,_ = np.linalg.lstsq(X.T,y)

predicted =  GoCs*x[0]+x[1]
Gene_minus_predicted = GeneCs-predicted

do_plot = False
if do_plot:
    p0,p1 = GoCs.min(),GoCs.max()
    plot([p0,p1],[np.dot(x,[p0,1]),np.dot(x,[p1,1])], 'b-')
    plot(GoCs.ravel(), GeneCs.ravel(),'r.')

values = []
for di,d in enumerate(sorted_drugs):
    for d2i,d2 in enumerate(sorted_drugs[di+1:]):
        if d in ["DMSO", "Media"] or d2 in ["DMSO","Media"]:
            continue
        values.append((d, d2, -Gene_minus_predicted[di,d2i]))
values.sort(key=lambda x:x[2])

def fix_dname(d):
    if d == 'H-7, Dihydrochloride': return 'H-7'
    if d == 'Doxorubicin hydrochloride': return 'Doxorubicin'
    return d

with open('final/DREAM7_DrugSensitivity2_Predictions_Lisbon.csv', 'w') as output:
    ac = -1
    print >>output, first_line
    for i,(d,d2,v) in enumerate(values):
        d = fix_dname(d)
        d2 = fix_dname(d2)
        print >> output, ('{0} & {1}, {2}'.format(d,d2, i +1))
        if ac == -1 and v > 0:
            ac = i
    print >> output, ('Compound pair with additive activity (IC36), {0} & {1}'.format(fix_dname(values[ac][0]),fix_dname(values[ac][1])))
