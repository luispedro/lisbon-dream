import numpy as np
from drugconcentrations import get_drugs
from load import read_sub2
data, drugs, times, concentrations, gene_names = read_sub2()
dmso = data[:,drugs == 'DMSO']
media = data[:,drugs == 'Media']
thresh_up_media = media.mean(1) + 2*media.std(1)
thresh_down_media = media.mean(1) - 2*media.std(1)
thresh_up_dmso = dmso.mean(1)+2*dmso.std(1)
thresh_down_dmso = dmso.mean(1)-2*dmso.std(1)

thresh_up = np.maximum(thresh_up_dmso, thresh_up_media)
thresh_down = np.minimum(thresh_down_dmso, thresh_down_media)

drugdata = get_drugs()
validinany = np.zeros(len(data), bool)
for dr in sorted(set(drugs)):
    ddata = data[:,drugs ==dr]
    dtimes = times[:,drugs == dr]
    valid = (ddata > thresh_up[:,None]) | (ddata < thresh_down[:,None])
    validinall = np.zeros(len(valid), bool)
    validinall |= valid[:,dtimes == 6].mean(1) > .5
    validinany |= validinall
    print '{0:32}{1: 8}'.format(dr, np.sum(validinall))
   
