all: mixture_models.pdf herschel_dust.pdf

MIXTURE_MODELS_FIGURES = \
	dev_K10_MR08_profile.png \
	dev_K10_MR08_profile_log.png \
	dev_K10_MR08_residual.png \
	dev_K10_MR08_residual_log.png

mixture_models.pdf: $(MIXTURE_MODELS_FIGURES)

%.png:
	curl -O "http://broiler.astrometry.net/~dwh2/Tractor/code/$@"

%.pdf: %.tex
	pdflatex $<
	bash -c " ( grep Rerun $*.log && pdflatex $< ) || echo noRerun "
	bash -c " ( grep Rerun $*.log && pdflatex $< ) || echo noRerun "
