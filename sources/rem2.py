import numpy as np
from drugconcentrations import get_drugs
from load import read_sub2
data, drugs, times, concentrations, gene_names = read_sub2()


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
    dmso = data[:,drugs == 'DMSO']
    media = data[:,drugs == 'Media']
    thresh_up_media = media.mean(1) + 2*media.std(1)
    thresh_down_media = media.mean(1) - 2*media.std(1)
    thresh_up_dmso = dmso.mean(1)+2*dmso.std(1)
    thresh_down_dmso = dmso.mean(1)-2*dmso.std(1)

    thresh_up = np.maximum(thresh_up_dmso, thresh_up_media)
    thresh_down = np.minimum(thresh_down_dmso, thresh_down_media)
    return thresh_up, thresh_down_dmso

thresh_up, thresh_down = threshold(data)

drugdata = get_drugs()
validinany = np.zeros(len(data), bool)
for dr in sorted(set(drugs)):
    ddata = data[:,drugs ==dr]
    dtimes = times[:,drugs == dr]
    valid = (ddata > thresh_up[:,None]) | (ddata < thresh_down[:,None])
    validinall = np.zeros(len(valid), bool)
    validinall |= valid[:,dtimes == 6].mean(1) > .5
    print '{0:32}{1: 8}'.format(dr, np.sum(validinall))
   
def retrieve_gos(names):
    import waldo
    import waldo.uniprot
    gos = []
    for name in names:
        e = waldo.uniprot.retrieve.retrieve_entry(name+'_HUMAN')
        if e is None:
            gos.append([])
        else:
            gos.append([ann.go_id for ann in e.go_annotations])
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
validinany = np.zeros(len(perg), bool)
for dr in sorted(set(drugs)):
    ddata = perg[:,drugs==dr]
    dtimes = times[:,drugs == dr]
    valid = (ddata > thresh_up[:,None]) | (ddata < thresh_down[:,None])
    validinall = np.zeros(len(valid), bool)
    validinall |= valid.mean(1) > .5
    print '{0:32}{1: 8}'.format(dr, np.sum(validinall))

