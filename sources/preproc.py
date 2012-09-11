from load import *

def nanmean(arr, axis=None):
    nancounts = np.sum(~np.isnan(arr), axis=axis)
    return np.nansum(arr,axis=axis)/nancounts


def zscore1(arr):
    from milk.unsupervised import zscore
    return zscore(arr, axis=1)

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


def ge_rna_valid():
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
    R = zscore1(R)
    GE = zscore1(GE)
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
            features.append(np.array(cur).mean(0))
            labels.append(training[ci])
    features = np.array(features)
    labels = np.array(labels)
    return features, labels, gene2ensembl

def rna_ge_gosweigths():
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
            gosweigths[:,ci] += f

    from milk.unsupervised import pdist
    dists = pdist(gosweigths.T)
    X,Y = np.where(dists == 0)
    select = np.ones(len(gosweigths.T), bool)
    for x,y in zip(X,Y):
        if x != y and select[x] and select[y]:
            select[x] = 0
    return gosweigths[:,select], labels
