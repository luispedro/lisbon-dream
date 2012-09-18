from load import read_methyl
def test_methyl():
    methyl,CpGs = read_methyl()
    assert methyl.shape[0] == len(CpGs)
