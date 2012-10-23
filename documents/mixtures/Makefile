all: mixture_models.pdf

MIXTURE_MODELS_FIGURES = \
	exp_K06_MR08_profile.pdf \
	exp_K06_MR08_profile_log.pdf \
	exp_K06_MR08_residual.pdf \
	exp_K06_MR08_fractional_log.pdf\
	dev_K10_MR08_profile.pdf \
	dev_K10_MR08_profile_log.pdf \
	dev_K10_MR08_residual.pdf \
	dev_K10_MR08_fractional_log.pdf \
	lux_K06_MR04_profile.pdf \
	lux_K06_MR04_profile_log.pdf \
	luv_K08_MR08_profile.pdf \
	luv_K08_MR08_profile_log.pdf \
	ser2_K10_MR08_profile_log.pdf \
	ser3_K10_MR08_profile_log.pdf \
	ser5_K10_MR08_profile_log.pdf \
	mixtures_vs_model_K10.pdf \
	mixtures_vs_K_exp.pdf \
	mixtures_vs_K_dev.pdf \
	mixtures_vs_K_lux.pdf \
	mixtures_vs_K_luv.pdf

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

%.pdf:
	curl -O "http://broiler.astrometry.net/~dwh2/Tractor/code/$@"
