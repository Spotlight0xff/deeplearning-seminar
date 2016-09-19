# Makefile for i6 slides demo
# by Philippe Dreuw and Thomas Deselaers, 2005
#
# $Id: Makefile,v 1.13 2007-11-15 07:58:53 dreuw Exp $
#
default: slides.pdf

overview: slides.pdf
	pdfnup --nup 2x4 --column true $^

handout: slides.pdf
	pdfnup --nup 2x2 --column true $^

hacks: slides.tex
	pdflatex -shell-escape "\nonstopmode\input{slides}"''
	bibtex slides
	pdflatex -shell-escape "\nonstopmode\input{slides}"''
	pdflatex -shell-escape "\nonstopmode\input{slides}"''

%.pdf: %.tex
	pdflatex "\nonstopmode\input{$<}"''
	bibtex $*
	pdflatex "\nonstopmode\input{$<}"''
	pdflatex "\nonstopmode\input{$<}"''

%.ps: %.dvi
	dvips $<

%.dvi: %.tex
	latex "\nonstopmode\input{$<}"''
	bibtex $*
	latex "\nonstopmode\input{$<}"''
	latex "\nonstopmode\input{$<}"''

all: 
	rm -f slides.pdf slides.ps slidesSection.pdf slidesSection.ps presentation_template.pdf presentation_template.ps
	make slides.pdf
	make slides.ps
	make slidesSection.pdf
	make slidesSection.ps
	make presentation_template.pdf
	make presentation_template.ps


clean:
	rm -f *.aux *~ *.out *.log *.bbl *.blg *.brf *.dvi *.toc *.mtc*

clean-all: clean
	rm -f slides.pdf slides.ps slidesSection.pdf slidesSection.ps presentation_template.pdf presentation_template.ps
