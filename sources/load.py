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

def _read_rnaseq(fname):
    import numpy as np
    input = open_data(fname, 1)
    data = []
    celltypes = input.readline()
    celltypes = celltypes.strip().split('\t')
    celltypes = celltypes[2:]
    genes = []
    for line in input:
        line = line.strip()
        tokens = line.split()
        genes.append(tokens[:2])
        data.append(map(float,tokens[2:]))

    gene_transcripts = np.array(data)
    return gene_transcripts, celltypes, genes

def read_rnaseq():
    return _read_rnaseq('DREAM7_DrugSensitivity1_RNAseq_quantification.txt')

def read_rnaseq_calls():
    return _read_rnaseq('DREAM7_DrugSensitivity1_RNAseq_expressed_calls.txt')

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


def read_methyl(filter_methyl=0):
    import numpy as np
    ifile = open_data('DREAM7_DrugSensitivity1_Methylation.txt', 1)
    headers = ifile.readline().strip().split('\t')
    headers =  headers[2:]
    data = [line.strip().split('\t') for line in ifile]
    methyl = np.array([map(float,d[4:]) for d in data])
    CpGs = np.array([int(d[2]) for d in data])
    methyl = methyl[CpGs >= filter_methyl,:]
    return headers,methyl,CpGs


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

def read_sub2():
    '''
    data, drugs, times, concentrations, gene_names = read_sub2()
    '''
    ifile = open_data('ACalifano_DLBCL_Ly3_14Comp_treatment.txt', 2)
    gene_names = ifile.readline().strip().split('\t')
    drugs = ifile.readline().strip().split('\t')
    drugs = drugs[2:]
    for i in xrange(len(drugs)):
        if drugs[i][0] == '"' and drugs[i][-1] == '"':
            drugs[i] = drugs[i][1:-1]
    drugs = np.array(drugs)
    times = ifile.readline().strip().split('\t')
    times = np.array(map(int, times[2:]))
    concentrations = ifile.readline().strip().split('\t')
    concentrations = np.array(concentrations[2:])

    data = []
    genes = []

    while True:
        line = ifile.readline()
        if not line:
            break
        tokens = line.strip().split('\t')
        genes.append((tokens[0], tokens[1]))
        data.append(map(float, tokens[2:]))

    data = np.array(data)
    return data, drugs, times, concentrations, genes
