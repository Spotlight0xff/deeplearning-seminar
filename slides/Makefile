# Makefile for i6 slides demo
# by Philippe Dreuw and Thomas Deselaers, 2005
#
# $Id: Makefile,v 1.13 2007-11-15 07:58:53 dreuw Exp $
#
FIGURES=$(wildcard figures/*.tex)
MAINTEX=presentation

.PHONY: $(FIGURES)

default: $(FIGURES) $(MAINTEX).pdf



$(FIGURES):
	pdflatex -interaction nonstopmode -output-directory figures/ $@

$(MAINTEX).pdf: $(FIGURES) $(MAINTEX).tex
	pdflatex "\nonstopmode\input{$(MAINTEX)}"''
	bibtex $(MAINTEX)
	pdflatex "\nonstopmode\input{$(MAINTEX)}"''
	pdflatex "\nonstopmode\input{$(MAINTEX)}"''

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
	rm -f slides.pdf slides.ps slidesSection.pdf slidesSection.ps presentation.pdf presentation.ps
	#rm -f $(FIGURES:.tikz=.pdf)
	make slides.pdf
	make slides.ps
	make slidesSection.pdf
	make slidesSection.ps
	make presentation.pdf
	make presentation.ps


clean:
	rm -f *.aux *~ *.out *.log *.bbl *.blg *.brf *.dvi *.toc *.mtc*
	rm -f $(FIGURES:.tex=.pdf)

clean-all: clean
	rm -f slides.pdf slides.ps slidesSection.pdf slidesSection.ps presentation.pdf presentation.ps
