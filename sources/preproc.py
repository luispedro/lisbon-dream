# -*- coding: utf-8 -*-
from load import *

def nanmean(arr, axis=None):
    nancounts = np.sum(~np.isnan(arr), axis=axis)
    return np.nansum(arr,axis=axis)/nancounts

def maxabs(arr):
    arr = np.asanyarray(arr)
    select = np.abs(arr).argmax(0)
    return np.array([arr[y,x] for x,y in enumerate(select)])

def normlabels(labels, axis=0):
    labels = np.array(labels)
    labels = labels.copy()
    if axis == 0:
        labels -= nanmean(labels, axis)
    elif axis == 1:
        labels -= nanmean(labels, axis)[:,None]
    else:
        raise ValueError('Axis ∉ { 0, 1}')

    return labels


def rna_ge_concatenated():
    gene_exp,celltypes_ge,_ = read_gene_expression()
    rna_seq,celltypes_rna,_ = read_rnaseq()
    training,celltypes,cs = read_training()
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

def rna_seq_active_only():
    rna_seq,celltypes_rna,rna_types = read_rnaseq()
    rna_seqcalls,rsc_cells, rsc_genes = read_rnaseq_calls()
    training,celltypes,cs = read_training()
    ptp = rna_seqcalls.ptp(1)
    valid = (ptp == 1)
    features = []
    labels = []
    for ci,ct in enumerate(celltypes):
        try:
            rid = celltypes_rna.index(ct)
            features.append(rna_seq[valid, rid])
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
            elif aggr in ('maxabs', 'max(abs)'):
                features.append(maxabs(cur))
            labels.append(training[ci])
    features = np.array(features)
    return features, np.array(labels), gene2ensembl

def rna_ge_gosweigths(ag='add'):
    import waldo
    from waldo import uniprot
    features,labels,gene2ensembl  = ge_rna_valid()
    gos = set()
    for g in gene2ensembl:
        uniprot_name = waldo.translate(gene2ensembl[g], 'ensembl:gene_id', 'uniprot:name')
        cur = uniprot.retrieve_go_annotations(uniprot_name, only_cellular_component=False)
        gos.update(cur)
    gos = list(gos)
    gosweigths = np.zeros((len(features), len(gos)), np.float)

    for f,g in zip(features.T, gene2ensembl.keys()):
        uniprot_name = waldo.translate(gene2ensembl[g], 'ensembl:gene_id', 'uniprot:name')
        cur = uniprot.retrieve_go_annotations(uniprot_name, only_cellular_component=False)
        for c in cur:
            ci = gos.index(c)
            if ag == 'add':
                gosweigths[:,ci] += f
            elif ag == 'maxabs':
                gosweigths[:,ci] += f * (np.abs(f) > np.abs(gosweigths[:, ci]))

    return prune_similar(gosweigths), np.array(labels)

def prune_similar(features, threshold=None, frac=None):
    from milk.unsupervised import pdist
    dists = pdist(features.T)
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
