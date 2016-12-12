TEX_FILES=$(shell find . -type f -name '*.tex')

paper.pdf: $(TEX_FILES)
	rm -f paper.sta
	pdflatex -shell-escape paper.tex
	pdflatex -shell-escape paper.tex
	pdflatex -shell-escape paper.tex
