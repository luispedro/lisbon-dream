from load import *
gene_exp,celltypes_ge,_ = read_gene_expression()
training,celltypes,cs = read_training()
labels = []
features = []
for ci,ct in enumerate(celltypes):
    try:
        id = celltypes_ge.index(ct)
        features.append(gene_exp[:,id])
        labels.append(training[ci])
    except ValueError:
        pass
    
features = np.array(features)
labels = np.array(labels)
labels[np.isnan(labels)] = 1+np.nanmax(labels)
