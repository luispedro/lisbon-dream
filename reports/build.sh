#!/usr/bin/zsh

latex=pdflatex
bibtex=bibtex

#latex=xelatex
#bibtex=biber


function build() {
    input=$1
    mkdir -p .$input.tex_files
    cd .$input.tex_files
    TEXINPUTS=.:..:../images/:../figures/: $latex $input
    BSTINPUTS=:..: BIBINPUTS=:..:.: $bibtex $input
    cp $input.pdf ..
    cd ..
}
build report

