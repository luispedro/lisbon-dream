from load import *

def nanmean(arr, axis=None):
    nancounts = np.sum(~np.isnan(arr), axis=axis)
    return np.nansum(arr,axis=axis)/nancounts

def preproc():
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
    labels = np.array(labels)
    labels -= nanmean(labels, 0)
    return features, labels

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
    labels = np.array(labels)
    labels -= nanmean(labels, 0)
    return features, labels
