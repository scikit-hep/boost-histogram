SRC = $(wildcard *.tex)
PDFs = $(patsubst %.tex, %.pdf, $(SRC))
PNGs = $(patsubst %.tex, %.png, $(SRC))

.PHONY: all

all: $(PNGs)

%.pdf: %.tex Makefile
	pdflatex $<
	@rm $*.aux $*.log

%.png: %.pdf
	convert -density 150 $< -quality 95 -colorspace Gray $@
