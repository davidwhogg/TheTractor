.SUFFIXES: png pdf tex

all: mixture_models.pdf herschel_dust.pdf

mixture_models.pdf: dev_K11_MR08_profile_log.png

%.png:
	curl -O "http://broiler.astrometry.net/~dwh2/Tractor/code/$@"

%.pdf: %.tex
	pdflatex $<
	bash -c " ( grep Rerun $*.log && pdflatex $< ) || echo noRerun "
	bash -c " ( grep Rerun $*.log && pdflatex $< ) || echo noRerun "
