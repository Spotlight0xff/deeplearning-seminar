TEX_FILES=$(shell find . -type f -name '*.tex')

paper.pdf: $(TEX_FILES)
	pdflatex -shell-escape paper.tex
