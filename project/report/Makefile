NY: clean all watch ignore-report add-report
FILE=report
TEX=$(FILE).tex
PDF=$(FILE).pdf
TEXMAKE=latexmk -pdf

all: $(PDF)

$(PDF):
	$(TEXMAKE) $(TEX)

watch:
	$(TEXMAKE) -pvc $(TEX)

clean:
	rm -rf *.aux *.bbl *.blg *dvi *.log *.out *.synctex.gz *.toc *.lot *.lof
