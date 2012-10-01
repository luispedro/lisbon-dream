# -*- coding: utf-8 -*-
from load import *
from jug import TaskGenerator

def nanmean(arr, axis=None):
    nancounts = np.sum(~np.isnan(arr), axis=axis)
    return np.nansum(arr,axis=axis)/nancounts

def maxabs(arr):
    arr = np.asanyarray(arr)
    select = np.abs(arr).argmax(0)
    return np.array([arr[y,x] for x,y in enumerate(select)])

def rna_ge_concatenated():
    gene_exp,celltypes_ge,_ = read_gene_expression()
    rna_seq,celltypes_rna,_ = read_rnaseq()
    training,celltypes,cs = read_training()
    rna_seq = np.log1p(rna_seq)
    labels = []
    features = []
    for ci,ct in enumerate(celltypes):
        try:
            gid = celltypes_ge.index(ct)
            rid = celltypes_rna.index(ct)
            features.append(np.hstack([gene_exp[:,gid], rna_seq[:,rid]]))
            labels.append(training[ci])
        except ValueError:
            pass
    features = np.array(features)
    return features, np.array(labels)

def ge_rna_valid(aggr='mean'):
    from milk.unsupervised import zscore
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
    features = []
    labels = []
    for ci,ct in enumerate(celltypes):
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
        if len(cur):
            if aggr == 'mean':
                features.append(np.array(cur).mean(0))
            elif aggr == 'maxabs':
                features.append(maxabs(cur))
            labels.append(training[ci])
    features = np.array(features)
    return features, np.array(labels), gene2ensembl

@TaskGenerator
def rna_ge_gosweigths(ag='add', filter_gos=None):
    import waldo
    from waldo import uniprot
    features,labels,gene2ensembl  = ge_rna_valid()
    def select_gos(gos):
        from waldo.go import vocabulary
        if filter_gos is None:
            return gos
        return [g for g in gos if vocabulary(g) in filter_gos]
    gos = set()
    for g in gene2ensembl:
        uniprot_name = waldo.translate(gene2ensembl[g], 'ensembl:gene_id', 'uniprot:name')
        cur = uniprot.retrieve_go_annotations(uniprot_name, only_cellular_component=False)
        gos.update(cur)
    gos = list(gos)
    gos = select_gos(gos)
    if len(gos) == 0:
        raise ValueError('Gos were empty with filter = {0}'.format(filter_gos))
    gosweigths = np.zeros((len(features), len(gos)), np.float)

    for f,g in zip(features.T, gene2ensembl.keys()):
        uniprot_name = waldo.translate(gene2ensembl[g], 'ensembl:gene_id', 'uniprot:name')
        cur = uniprot.retrieve_go_annotations(uniprot_name, only_cellular_component=False)
        cur = select_gos(cur)
        for c in cur:
            ci = gos.index(c)
            if ag == 'add':
                gosweigths[:,ci] += f
            elif ag == 'maxabs':
                gosweigths[:,ci] = maxabs([f,gosweigths[:, ci]])
            else:
                raise ValueError('What did you mean ({0})'.format(ag))

    return gosweigths, np.array(labels)

def prune_similar(features, threshold=None, frac=None):
    from milk.unsupervised import pdist
    dists = pdist(features.T.astype(np.float32))
    if frac is not None:
        Ds = dists.ravel().copy()
        Ds.sort()
        threshold = Ds[int(frac*len(Ds))]
    elif threshold is None:
        threshold = 0
    X,Y = np.where(dists <= threshold)
    select = np.ones(len(features.T), bool)
    for x,y in zip(X,Y):
        if x != y and select[x] and select[y]:
            select[x] = 0
    return features[:,select]
