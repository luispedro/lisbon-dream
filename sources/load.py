import numpy as np
def read_gene_expression():
    input = open('DREAM7_DrugSensitivity1_GeneExpression.txt')
    data = []
    celltypes = input.readline()
    celltypes = celltypes.strip().split('\t')
    celltypes = celltypes[1:]

    genes = []
    for line in input:
        line = line.strip()
        tokens = line.split()
        genes.append(tokens[0])
        data.append(map(float,tokens[1:]))
        
    gene_expression = np.array(data)
    return gene_expression, celltypes, genes

def read_training():
    input = open('DREAM7_DrugSensitivity1_Drug_Response_Training.txt')
    header = input.readline()
    celltypes = header.strip().split('\t')
    celltypes = celltypes[1:]
    def asf(x):
        if x == 'NA': return float('NaN')
        return float(x)
    drugs = []
    drugvalues = []
    for line in input:
        tokens = line.strip().split('\t')
        drugs.append(tokens[0])
        drugvalues.append(map(asf,tokens[1:]))
        
    return np.array(drugvalues)
