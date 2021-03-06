from selectlearner import corrcoefs
import numpy as np
from drugconcentrations import get_drugs
from load import read_sub2
from matplotlib import pyplot as plt
import jug.utils
from jug import Task, TaskGenerator, CachedFunction, bvalue

reverse_regression = False

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


def fix_dname(d):
    if d == 'H-7, Dihydrochloride': return 'H-7'
    if d == 'Doxorubicin hydrochloride': return 'Doxorubicin'
    return d
def retrieve_gos(names, vocabs):
    import waldo
    import waldo.uniprot
    from waldo.go import vocabulary
    from waldo.backend import create_session
    session = create_session()
    vocccache = {}
    def cvocabulary(g):
        try:
            return vocccache[g]
        except KeyError:
            v = vocabulary(g, session=session)
            vocccache[g] = v
            return v
    gos = []
    for name in names:
        e = waldo.uniprot.retrieve.retrieve_entry(name+'_HUMAN', session=session)
        if e is None:
            gos.append([])
        else:
            anns = [ann.go_id for ann in e.go_annotations]
            if vocabs is not None:
                anns = [a for a in anns if cvocabulary(a) in vocabs]
            gos.append(anns)
    return gos


@TaskGenerator
def build_gos_pergos(data, names, vocabs):
    gos = retrieve_gos(names, vocabs)
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


data, drugs, times, concentrations, gene_names = read_sub2()
first_line = 'Drug Combination,Rank'


names = np.array([n for _,n in gene_names])
gene_names = np.array(gene_names)
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
idata = jug.utils.identity(data)

thresh_up, thresh_down = threshold(data)

drugdata = get_drugs()
valid_data = []

sorted_drugs = sorted(set(drugs))

genes_used = set()
for dr in sorted_drugs:
    ddata = data[:,drugs ==dr]
    dtimes = times[:,drugs == dr]
    valid = (ddata > thresh_up[:,None]) | (ddata < thresh_down[:,None])
    validinany = np.zeros(len(valid), bool)
    validinany |= valid[:].mean(1) > .5
    valid_data.append(validinany)
    print '{0:32}{1: 8}'.format(dr, np.sum(validinany))

    for g in gene_names[selected][validinany,1]:
        genes_used.add((dr, g))


with open('genes_used.txt', 'w') as outg:
    line = ['']
    for dr in sorted_drugs:
        if dr != 'DMSO':
            line.append(dr)
    print >>outg, '\t'.join(line)
    for g in gene_names[selected][:,1]:
        line = [g]
        print_line = False
        for dr in sorted_drugs:
            if dr == 'DMSO':
                continue
            if (dr, g) in genes_used:
                line.append('*')
                print_line = True
            else:
                line.append('')
        if print_line:
            print >>outg, '\t'.join(line)

mf_valid_perg = None
for name,vocabs in [
                ('mf', ['molecular_function']),
                ('cc', ['cellular_component']),
                ('bp', ['biological_process']),
                ('mf_bp', ['molecular_function', 'biological_process']),
                ('mf_cc', ['molecular_function', 'cellular_component']),
                ('bp_cc', ['biological_process', 'cellular_component']),
                ('all',None),
                ]:
    allgos, perg = bvalue(build_gos_pergos(idata,names, vocabs))

    thresh_up, thresh_down = threshold(perg)
    valid_perg = []
    gos_used = set()
    for dr in sorted_drugs:
        ddata = perg[:,drugs==dr]
        dtimes = times[:,drugs == dr]
        valid = (ddata > thresh_up[:,None]) | (ddata < thresh_down[:,None])
        validinany = np.zeros(len(valid), bool)
        validinany |= valid.mean(1) > .5
        valid_perg.append(validinany)
        print '{0:32}{1: 8}'.format(dr, np.sum(validinany))
        for i,g in enumerate(allgos):
            if validinany[i]:
                gos_used.add((dr, g))

    if mf_valid_perg is None and name == 'mf':
        with open('goterms_used.txt', 'w') as outg:
            line = ['']
            for dr in sorted_drugs:
                if dr != 'DMSO':
                    line.append(dr)
            print >>outg, '\t'.join(line)
            for g in allgos:
                line = [g]
                print_line = False
                for dr in sorted_drugs:
                    if dr == 'DMSO':
                        continue
                    if (dr, g) in gos_used:
                        line.append('*')
                        print_line = True
                    else:
                        line.append('')
                if print_line:
                    print >>outg, '\t'.join(line)
        mf_valid_perg = np.array(valid_perg)


    GeneCs = np.array([corrcoefs(valid_data, v) for v in valid_data])
    GoCs = np.array([corrcoefs(valid_perg, p) for p in valid_perg])

    if reverse_regression:
        y = GoCs.ravel()
        X = GeneCs.ravel()
    else:
        X = GoCs.ravel()
        y = GeneCs.ravel()
    y = y[X.ravel() <= .99]
    X = X[X <= .99]
    X = np.array([X,np.ones(len(X))])
    x,r,_,_ = np.linalg.lstsq(X.T,y)

    if reverse_regression:
        predicted =  GeneCs*x[0]+x[1]
        Gene_minus_predicted = -GoCs+predicted
    else:
        predicted =  GoCs*x[0]+x[1]
        Gene_minus_predicted = GeneCs-predicted

    values = []
    for di,d in enumerate(sorted_drugs):
        for d2i,d2 in enumerate(sorted_drugs[di+1:]):
            if d in ["DMSO", "Media"] or d2 in ["DMSO","Media"]:
                continue
            values.append((d, d2, -Gene_minus_predicted[di,di+1+d2i]))
    values.sort(key=lambda x:x[2])

    end = ('r' if reverse_regression else '')
    with open('outputs/sub2{}_{}.csv'.format(end,name), 'w') as output:
        ac = -1
        print >>output, first_line
        for i,(d,d2,v) in enumerate(values):
            d = fix_dname(d)
            d2 = fix_dname(d2)
            print >> output, ('{0} & {1}, {2}'.format(d,d2, i +1))
            if ac == -1 and v > 0:
                ac = i
        print >> output, ('Compound pair with additive activity (IC36), {0} & {1}'.format(fix_dname(values[ac][0]),fix_dname(values[ac][1])))


    plt.clf()
    if reverse_regression:
        plt.plot(GoCs.ravel(), GeneCs.ravel(), 'ko')
        plt.ylabel('Gene Correlation')
        plt.xlabel('GO Term Correlation')
        xline = np.array([GoCs.min()-.1, GoCs.max()+.1])
    else:
        plt.plot(GeneCs.ravel(), GoCs.ravel(), 'ko')
        plt.xlabel('Gene Correlation')
        plt.ylabel('GO Term Correlation')
        X = GeneCs.ravel()
        y = GoCs.ravel()
        y = y[X.ravel() <= .99]
        X = X[X <= .99]
        X = np.array([X,np.ones(len(X))])
        x,r,_,_ = np.linalg.lstsq(X.T,y)
        xline = np.array([GeneCs.min()-.1, GeneCs.max()+.1])
    yline = xline*x[0] + x[1]
    plt.plot(xline, yline, 'r-')
    plt.savefig('outputs/plot{}_{}.png'.format(end, name), dpi=600)

