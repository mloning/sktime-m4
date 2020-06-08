# Makefile for easier installation and cleanup.
#
# Uses self-documenting macros from here:
# http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
# SHELL:=/bin/zsh

TIME=`date +'%y.%m.%d_%H:%M:%S'`

.PHONY: help cover dist venv
.DEFAULT_GOAL := help


help:
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) |\
		 awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m\
		 %s\n", $$1, $$2}'

.ONESHELL:
update:
	# update repo
	git pull

	# update dev installs
	cd ~/Documents/Research/toolboxes/sktime/
	git checkout fcast
	git pull

	cd ~/Documents/Research/toolboxes/sktime-dl/
	git checkout dev
	git pull

prepare:
	python ./scripts/prepare_original_results.py

replicate:
	nohup python -u ./scripts/replicate.py > run_$(TIME).out &

evaluate:
	python ./scripts/evaluate.py

tables:
	python ./scripts/make_tables.py