from collections import namedtuple

def get_drugs():
	"""
	Returns
	-------
	drugs : array of objects
		List of drug concentrations. Each element has the following properties:
		well : string
		drug : string
		concentration : float (in uM)
		timeAtIC20 : int {24, 48 } (in hours)
	"""
	DrugConcentration = namedtuple('DrugConcentration', ('well', 'drug', 'concentration', 'timeAtIC20'))

	return [
		DrugConcentration('C09', 'Aclacinomycin A', 0.105, 24),
		DrugConcentration('D12', 'Aclacinomycin A', 0.036, 48),
		DrugConcentration('A09', 'Blebbistatin', 100., 24),
		DrugConcentration('B12', 'Blebbistatin',  10., 48),
		DrugConcentration('B03', 'Camptothecin', 0.31, 24),
		DrugConcentration('C06', 'Camptothecin', 0.01, 48),
		DrugConcentration('C03', 'Cycloheximide', 5., 24),
		DrugConcentration('D06', 'Cycloheximide', 0.264, 48),
		DrugConcentration('B09', 'Doxorubicin hydrochloride', 0.101, 24),
		DrugConcentration('C12', 'Doxorubicin hydrochloride', 0.03, 48),
		DrugConcentration('D03', 'Etoposide', 0.811, 24),
		DrugConcentration('E06', 'Etoposide', 0.812, 48),
		DrugConcentration('E12', 'Geldanamycin', 0.001, 48),
		DrugConcentration('D09', 'Geldanamycin', 0.032, 24),
		DrugConcentration('H10', 'H-7, Dihydrochloride', 12.4, 48),
		DrugConcentration('H09', 'H-7, Dihydrochloride', 20.2, 24),
		DrugConcentration('B06', 'Methotrexate', 100., 48),
		DrugConcentration('A03', 'Methotrexate', 100., 24),
		DrugConcentration('F06', 'Mitomycin C', 0.553, 48),
		DrugConcentration('F03', 'Mitomycin C', 2.56, 24),
		DrugConcentration('H06', 'Monastrol',  50., 48),
		DrugConcentration('H03', 'Monastrol', 100., 24),
		DrugConcentration('F09', 'Rapamycin',  22., 24),
		DrugConcentration('F12', 'Rapamycin', 13.8, 48),
		DrugConcentration('F01', 'Trichostatin A', 0.143, 24),
		DrugConcentration('G06', 'Trichostatin A', 0.114, 48),
		DrugConcentration('F07', 'Vincristine', 0.01, 24),
		DrugConcentration('G12', 'Vincristine', 0.005, 48),
        ]

