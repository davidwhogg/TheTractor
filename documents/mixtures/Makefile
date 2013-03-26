all: mixture_models.pdf

FIG_SUFFIX = pdf

MIXTURE_MODELS_FIGURES = \
	exp_K06_MR08_profile.$(FIG_SUFFIX) \
	exp_K06_MR08_profile_log.$(FIG_SUFFIX) \
	exp_K06_MR08_residual.$(FIG_SUFFIX) \
	exp_K06_MR08_fractional_log.$(FIG_SUFFIX)\
	dev_K10_MR08_profile.$(FIG_SUFFIX) \
	dev_K10_MR08_profile_log.$(FIG_SUFFIX) \
	dev_K10_MR08_residual.$(FIG_SUFFIX) \
	dev_K10_MR08_fractional_log.$(FIG_SUFFIX) \
	lux_K06_MR04_profile.$(FIG_SUFFIX) \
	lux_K06_MR04_profile_log.$(FIG_SUFFIX) \
	luv_K08_MR08_profile.$(FIG_SUFFIX) \
	luv_K08_MR08_profile_log.$(FIG_SUFFIX) \
	ser2_K10_MR08_profile_log.$(FIG_SUFFIX) \
	ser3_K10_MR08_profile_log.$(FIG_SUFFIX) \
	ser5_K10_MR08_profile_log.$(FIG_SUFFIX) \
	mixtures_vs_model_K10.$(FIG_SUFFIX) \
	mixtures_vs_K_exp.$(FIG_SUFFIX) \
	mixtures_vs_K_dev.$(FIG_SUFFIX) \
	mixtures_vs_K_lux.$(FIG_SUFFIX) \
	mixtures_vs_K_luv.$(FIG_SUFFIX) \
	trdemo-01.$(FIG_SUFFIX) \
	trdemo-02.$(FIG_SUFFIX) \
	trdemo-03.$(FIG_SUFFIX) \
	trdemo-06.$(FIG_SUFFIX) \
	trdemo-08.$(FIG_SUFFIX) \
	trdemo-09.$(FIG_SUFFIX)

MIXTURE_MODELS_TABLES = \
	table_exp.tex \
	table_dev.tex \
	table_lux.tex \
	table_luv.tex # \
	table_ser2.tex \
	table_ser3.tex \
	table_ser5.tex \

mixture_models.pdf: $(MIXTURE_MODELS_FIGURES) $(MIXTURE_MODELS_TABLES) mixture_models.tex
	pdflatex mixture_models
	bash -c " ( grep Rerun mixture_models.log && pdflatex mixture_models ) || echo noRerun "
	bash -c " ( grep Rerun mixture_models.log && pdflatex mixture_models ) || echo noRerun "

table_%.tex:
	curl -O "http://broiler.astrometry.net/~dwh2/Tractor/code/$@"

%.eps: %.pdf
	convert -density 300 $< $@

%.pdf:
	curl -O "http://broiler.astrometry.net/~dwh2/Tractor/code/$@"
