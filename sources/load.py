import numpy as np

def open_data(fname, subchallenge):
    from os import path
    return open(path.join(
                path.dirname(path.abspath(__file__)),
                '..',
                'data',
                'DrugSensitivity%s' % subchallenge,
                fname))

def read_gene_expression():
    input = open_data('DREAM7_DrugSensitivity1_GeneExpression.txt', 1)
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
    input = open_data('DREAM7_DrugSensitivity1_Drug_Response_Training.txt', 1)
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
        
    return np.array(drugvalues), celltypes, drugs
