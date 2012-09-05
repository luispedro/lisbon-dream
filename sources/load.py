import numpy as np

def open_data(fname, subchallenge):
    '''
    file_obj = open_data(fname, subchallenge)

    Get input data (as an open file) for subchallenge ``subchallenge``
    '''
    from os import path
    return open(path.join(
                path.dirname(path.abspath(__file__)),
                '..',
                'data',
                'DrugSensitivity%s' % subchallenge,
                fname))

def read_rnaseq():
    import numpy as np
    input = open_data('DREAM7_DrugSensitivity1_RNAseq_quantification.txt', 1)
    data = []
    celltypes = input.readline()
    celltypes = celltypes.strip().split('\t')
    celltypes = celltypes[1:]
    genes = []
    for line in input:
        line = line.strip()
        tokens = line.split()
        genes.append(tokens[:2])
        data.append(map(float,tokens[2:]))

    gene_transcripts = np.array(data)
    return gene_transcripts, celltypes, genes

def read_gene_expression():
    '''
    gene_expression, celltypes, genes = read_gene_expression()
    '''
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
    '''
    drugvalues, celltypes, drugs = read_training()
    '''
    input = open_data('DREAM7_DrugSensitivity1_Drug_Response_Training.txt', 1)
    header = input.readline()
    drugs = header.strip().split('\t')
    drugs = drugs[1:]
    def asf(x):
        if x == 'NA': return float('NaN')
        return float(x)

    drugvalues = []
    celltypes = []
    for line in input:
        tokens = line.strip().split('\t')
        celltypes.append(tokens[0])
        drugvalues.append(map(asf,tokens[1:]))
        
    return np.array(drugvalues), celltypes, drugs
