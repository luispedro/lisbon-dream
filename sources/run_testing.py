from scipy import stats
from load import *
from norm_learners import norm_learner
from milk.supervised.normalise import zscore_normalise
from milk.supervised.classifier import ctransforms
from selectlearner import *
from regularized import *
from milk.unsupervised import zscore
from preproc import *
from load import *
first_line = 'DrugAnonID,Drug1,Drug2,Drug3,Drug4,Drug5,Drug6,Drug7,Drug8,Drug9,Drug10,Drug11,Drug12,Drug13,Drug14,Drug15,Drug16,Drug17,Drug18,Drug19,Drug20,Drug21,Drug22,Drug23,Drug24,Drug25,Drug26,Drug27,Drug28,Drug29,Drug30,Drug31'
def read_olines():
    ifile = open_data('DREAM7_DrugSensitivity1_Predictions.csv', 1)
    ifile.readline().split(',')[0]
    olines = [ifile.readline().split(',')[0] for i in xrange(53)]
    assert not ifile.readline()
    return olines
rna_seq,celltypes_rna,rna_types = read_rnaseq()
gene_exp,celltypes_ge,ge_genes = read_gene_expression()
rna_seqcalls,rsc_cells, rsc_genes = read_rnaseq_calls()
rna_seq = np.log1p(rna_seq)
valid = rna_seqcalls.ptp(1) == 1
rsc_genes = np.array(rsc_genes)
gene2ensembl = {g:e for g,e in rsc_genes[valid]}
GE = []
R = []
r_genes = [g for g,_ in rsc_genes]
for g in gene2ensembl:
     try:
        gi = ge_genes.index(g)
        ri = r_genes.index(g)
        GE.append(gene_exp[gi])
        R.append(rna_seq[ri])
     except:
        pass
R = np.array(R)
GE = np.array(GE)
R = zscore(R, axis=1)
GE = zscore(GE, axis=1)
training,celltypes,cs = read_training()
allct = set(celltypes_ge)|set(celltypes_rna)
testing = allct - set(celltypes)
features = []
selected = []
ocelltypes = []
allct = list(allct)
for ct in allct:
    selected.append(ct in testing)
    if ct in testing:
        ocelltypes.append(ct)
    cur = []
    try:
        gi = celltypes_ge.index(ct)
        cur.append(GE[:,gi])
    except ValueError:
        pass
    try:
        ri = celltypes_rna.index(ct)
        cur.append(R[:,ri])
    except ValueError:
        pass
    if not len(cur):
        raise KeyError("?")
    features.append(np.array(cur).mean(0))

selected = np.array(selected)
features = np.array(features)
gosweights = generate_gosweights(features, gene2ensembl, 'maxabs', ['molecular_function', 'biological_process'])
gosweights = thresh_features(gosweights)



learner = norm_learner(ctransforms(remove_constant_features(), zscore_normalise(), select_learner(12), lasso_relaxed(.000225010113525, .1)), 0)
rna_ge_gosweigths_mpbf_ma,labels = rna_ge_gosweigths.f('maxabs', ['molecular_function', 'biological_process'])
rna_ge_gosweigths_mpbf_ma = thresh_features(rna_ge_gosweigths_mpbf_ma)
model = learner.train(rna_ge_gosweigths_mpbf_ma, labels)



drugvalues, celltypes, drugs = read_training()
olines = read_olines()
results = []
for o in olines:
    if o in testing:
        oi = allct.index(o)
        r = model.apply(gosweights[oi])
    else:
        oi = celltypes.index(o)
        r = drugvalues[oi]
    results.append(r)
results = np.array(results)

def rankint(ri):
    ri = np.array(ri)
    # Values are switched around
    #" lowest ranking (1,2,3...) corresponds to the highest GI50 values "
    # therefore, we switch signs here:
    ri = -ri
    ri = ri.argsort()
    rri = np.zeros(len(ri))
    for i,r in enumerate(ri):
        rri[r] = i
    rri += 1
    return rri


rresults = np.array([rankint(ri) for ri in results.T]).T
rresults = rresults.astype(int)
with open('final/DREAM7_DrugSensitivity1_Predictions_Lisbon.csv', 'w') as output:
    print >>output, first_line
    for o,r in zip(olines, rresults):
        print >>output, ",".join(map(format,[o]+list(r)))
