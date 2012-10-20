all: mixture_models.pdf

MIXTURE_MODELS_FIGURES = \
	exp_K06_MR08_profile.png \
	exp_K06_MR08_profile_log.png \
	exp_K06_MR08_residual.png \
	exp_K06_MR08_fractional_log.png\
	dev_K10_MR08_profile.png \
	dev_K10_MR08_profile_log.png \
	dev_K10_MR08_residual.png \
	dev_K10_MR08_fractional_log.png \
	lux_K06_MR04_profile.png \
	lux_K06_MR04_profile_log.png \
	luv_K08_MR08_profile.png \
	luv_K08_MR08_profile_log.png \
	ser2_K10_MR08_profile_log.png \
	ser3_K10_MR08_profile_log.png \
	ser5_K10_MR08_profile_log.png \
	mixtures_vs_model_K10.png \
	mixtures_vs_K_exp.png \
	mixtures_vs_K_dev.png \
	mixtures_vs_K_lux.png \
	mixtures_vs_K_luv.png

MIXTURE_MODELS_TABLES = \
	table_exp.tex \
	table_dev.tex \
	table_lux.tex \
	table_luv.tex # \
	table_ser2.tex \
	table_ser3.tex \
	table_ser5.tex \

mixture_models.pdf: $(MIXTURE_MODELS_FIGURES) $(MIXTURE_MODELS_TABLES)

table_%.tex:
	curl -O "http://broiler.astrometry.net/~dwh2/Tractor/code/$@"

%.png:
	curl -O "http://broiler.astrometry.net/~dwh2/Tractor/code/$@"

%.pdf: %.tex
	pdflatex $<
	bash -c " ( grep Rerun $*.log && pdflatex $< ) || echo noRerun "
	bash -c " ( grep Rerun $*.log && pdflatex $< ) || echo noRerun "
