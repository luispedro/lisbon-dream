from milk.unsupervised import zscore
from preproc import *
from load import *
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
for ct in testing:
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

features = np.array(features)
gosweights = generate_gosweights(features, gene2ensembl, 'maxabs', ['molecular_function'])


