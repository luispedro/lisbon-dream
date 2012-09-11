
_url = "http://www.genenames.org/cgi-bin/hgnc_downloads.cgi?title=HGNC+output+data&hgnc_dbtag=onlevel=pri&=on&order_by=gd_app_sym_sort&limit=&format=text&.cgifields=&.cgifields=level&.cgifields=chr&.cgifields=status&.cgifields=hgnc_dbtag&&status=Approved&status=Entry+Withdrawn&status_opt=2&submit=submit&col=gd_hgnc_id&col=gd_app_sym&col=gd_app_name&col=gd_status&col=gd_prev_sym&col=gd_aliases&col=gd_pub_chrom_map&col=gd_pub_acc_ids&col=gd_pub_refseq_ids&where=gd_locus_group+%3D+'protein-coding%20gene'"

def retrieve():
    from os import system
    system('wget "{0}" -O genenames.tsv'.format(_url))

def read_genenames():
    gene2acc = {}
    gene2refseq = {}
    ifile = open('genenames.tsv')
    ifile.readline().strip().split('\t')
    for line in ifile:
        tokens = line.strip().split('\t')
        symbol = tokens[1]
        prev = tokens[4]
        syn = tokens[5]
        accs = tokens[7] if len(tokens) > 7 else ""
        refseqs = tokens[8] if len(tokens) > 8 else ""
        allnames = [symbol]
        if prev:
            allnames.extend(prev.split(', '))
        if syn:
             allnames.extend(syn.split(', '))
        for n in allnames:
             if accs:
                 for ac in accs.split(', '):
                     gene2acc[n] = ac
             if refseqs:
                 for r in refseqs.split(', '):
                     gene2refseq[n] = r
    return gene2acc, gene2refseq
