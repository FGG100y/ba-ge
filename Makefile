##############################################################################
# GLOBALS 全局变量                                                           #
##############################################################################
# src/ layout: 替代 'pip install -e .'
export PYTHONPATH = $(PWD)/src/

# 进入 pdb 使用 ipython 环境:
export PYTHONBREAKPOINT = IPython.core.debugger.set_trace
# export PYTHONBREAKPOINT = IPython.terminal.debugger.set_trace

##############################################################################
# ==项目环境==	                                                             #
#																			 #
# 从代码仓库克隆到本地之后，使用 `make install` 启动并创建项目运行环境 		 #
# 从代码仓库克隆到本地之后，使用 `make init_projectdata` 同步依赖数据集		 #
#																			 #
# 如果平台系统为 Windows 则先行安装 Git-for-Windows (推荐), 在其中运行本项目 #
#																			 #
# requirements.txt 文件用于保证所有新克隆出来的项目具有完整的运行环境依赖包  #
##############################################################################

USING_PYENV := true
WORK_OS_UNIX := true
RUN_IN_DOCKER := false
# LOCAL_NETWORK := false

# Virtual Environment directory name in the Container
# Keep consist to VIRTUAL_ENV=/opt/venv in Dockerfile
ifeq ($(RUN_IN_DOCKER),true)
	VENV := /opt/venv
endif
INSTALL_STAMP := $(VENV)/.install.stamp
# INSTALL_STAMP as the indicator file for updated status

ifeq ($(WORK_OS_UNIX),true)
	PY3 := $(shell command -v python3 2> /dev/null)
	ifeq ($(USING_PYENV),true)
		PYTHON := python
	else
		PYTHON := $(VENV)/bin/python
	endif
else
	PY3 := python
	PYTHON := $(VENV)/Scripts/python
endif

# Simple Explaination:
# `echo ${VAR-val}` will show the content of '$VAR' and defaults to 'val' if undefined
# (and double the $ for escaping)
#
# `command -v` is roughly the equivalent of `which` but built-in the shell

## Create virtual env and install from requirements.txt
install: $(INSTALL_STAMP)
$(INSTALL_STAMP): $(PYTHON) requirements.txt
	$(PYTHON) -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -U pip
	$(PYTHON) -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
	touch $(INSTALL_STAMP)

# Create Python Virtual Environment (default name: .venv)
# if PY3 not exists, exit (program end) else create .venv
$(PYTHON):
	@if [ -z $(PY3) ]; then echo "python3 could not be found."; exit 2; fi
	$(PY3) -m venv $(VENV)

# GDALCFG := $(shell command -v gdal-config 2> /dev/null)
# $(GDAL):
#     @if [ -z $(GDALCFG) ]; then echo "gdal-config could not be found. Try: apt install libgdal-dev"; exit 2; fi
#     apt install -y libgdal-dev

# # $(warning now reached end of ifeq, PY3=$(PY3))

PYVERSION := $(shell command python -V 2> /dev/null)
## Makefile debugging by printing messages
project_info:
	@echo 'project root: $(PYTHONPATH)'
	@echo 'python version: $(PYVERSION)'
	@echo 'python venv dir: $(VENV)'
	@echo 'program run on UNIX: $(WORK_OS_UNIX)'
	@echo 'program run in Container: $(RUN_IN_DOCKER)'

##############################################################################
# ==项目规则==	                                                             #
# PROJECT RULES                                                              #
# ==项目规则==	                                                             #
#																			 #
# NOTE: This Makefile was also designed for debugging						 #
#																			 #
# .PHONY 任务																 #
# 	不依赖"任务文件"更新状态,												 #
#   例如：当前存在一个命名为 run_alltask 的文件, make 照样运行这个任务		 #
#																			 #
##############################################################################
.PHONY: app_run
## RUN THE PROJECT APP >>> ALL IN ONE GO
app_run:
	$(PYTHON) src/app.py

.PHONY: harmony
## Check src/ layout for importing module (temporary check of the scaffold)
harmony:
	$(PYTHON) src/cythonpkg/harmony.py

.PHONY: wakeup
## Wake the bot up
wakeup:
	$(PYTHON) src/wake_word/wake_gamgin_stream.py

.PHONY: query_LLM
## Query the LLM
query_LLM:
	$(PYTHON) src/nlp/llm_model.py

# .PHONY: stt_whisper_sr
# ## STT using whisper (speech_recognition)
# stt_whisper_sr:
#     $(PYTHON) src/listen_microphone.py

.PHONY: tts_speak
## TTS using gTTs or coqui_xtts-v2 (huggingface model)
tts_speak:
	$(PYTHON) src/audio/tts_model.py

.PHONY: stt_whisper_webui
## STT using whisper on webui
stt_whisper_webui:
	streamlit run src/audio/stream_webrtc_stt_whisper.py


# 数据库访问密码的加密；
# 重要：不清楚此模块的，请先阅读此模块
# .PHONY: generate_kek
# KEK of dbaccessing
# generate_kek:
#     $(PYTHON) ./src/data/dbio/key_management.py


# .PHONY: init_projectdata
# Initialize project dependent datasets
# init_projectdata: sync_data rebuild_middle_datasets


# ============================================================================
# NOTE SECTION ONE-------***DATA POOLING & CLEANING***---------------------{{{
# ============================================================================
# MIDDLE_DATA := ./data/interim
# RESULT_DATA := ./data/processed
#
#
# # Assume data and repo are on server
# USER := fmh
# # LOCAL_NETWORK := false
# ifeq ($(LOCAL_NETWORK),false)
#     HOSTNAME := gddst.wicp.vip
#     PORT := 2221
# else
#     HOSTNAME := 172.16.0.183
#     PORT := 22
# endif
#
# .PHONY: sync_data
# ## DEPENDENT DATA SYNCHRONIZATION
# sync_data: $(RSYNC)
#     rsync -azv --exclude 'tmp*' -e 'ssh -p $(PORT)' $(USER)@$(HOSTNAME):/home/$(USER)/dataGroup/projects/gzhzz/src/config ./src/
#     rsync -azv --exclude 'tmp*' -e 'ssh -p $(PORT)' $(USER)@$(HOSTNAME):/home/$(USER)/dataGroup/projects/gzhzz/data .
#
# HAS_RSYNC := $(shell command -v rsync 2> /dev/null)
# $(RSYNC):
#     @if [ -z $(HAS_RSYNC) ]; then echo "command rsync could not be found. Try: apt install rsync"; exit 2; fi
#     apt install -y rsync


# ============================================================================
# NOTE SECTION TWO-------***DATA MODELING & PREDICTION***---------------------
# ============================================================================
# .PHONY: run_modeltask
# ## MODLE TRAINING, PREDICTION & MAKE THE FINAL RESULT
# run_modeltask: model_train model_predict final_result model_performance

# =============================================================
# * train classifier
# * make prediction
# * result IO (DataBase)
# =============================================================

# RUN_BICLFIER := 1
# ifeq ($(RUN_BICLFIER),1)
#     LABELSYS := wqc_2cls
# else
#     LABELSYS := wqc_lmh
# endif
#
# DNN_TO_PREDICT := 1
# ifeq ($(DNN_TO_PREDICT),2)
#     CLFIER_ACHI := tfk
# else ifeq ($(DNN_TO_PREDICT),1)
#     CLFIER_ACHI := xgb
# else
#     CLFIER_ACHI := skl
# endif
#
# N_TREES := 1000

# .PHONY: model_train
# Train classification model
# model_train:
#     rm -f ./data/interim/reengineering_ssl_added_samples.csv
#     @echo 'delete file: ./data/interim/reengineering_ssl_added_samples.csv'
#     $(PYTHON) ./src/models/train_model.py --labelsys $(LABELSYS) --n_estimators $(N_TREES) -v 2


##############################################################################
# NOTE COMMANDS 普通命令                                                     #
##############################################################################

# =============================================================
# .PHONY 任务
# 	不依赖"任务文件"更新状态,
#   例如：当前存在一个命名为 clean 的文件, make 照样运行这个任务
# =============================================================

.PHONY: clean
## Delete all compiled Python files
clean:
	rm -f -r build/
	find . -type f -name "*.so" -delete
	find . -type f -name "*.py[co]" -delete

.PHONY: clean-cpp
## Delete all C/CPP file (generated by compiled extension)
clean-cpp:
	find . -type f -name "*.c" -delete
	find . -type f -name "*.cpp" -delete

.PHONY: build-ext
## Build extension by setup.py
build-ext: clean
	python setup.py build_ext --inplace

.PHONY: cython-build-ext
## Build extension by setup.py using Cython
cython-build-ext: clean clean-cpp
	python setup.py build_ext --inplace --use-cython

.PHONY: wheels
## Build App's distributions (pyproject-build)
wheels: clean clean-cpp
	python -m build


.PHONY: freeze
## Generate project's requirements file
freeze:
	$(PYTHON) -m pip freeze | grep -v "pkg-resources" > requirements.txt

# find . -type d -name "__pycache__" -delete
.PHONY: clean-all
## Delete all compiled Python files and remove venv
clean-all: clean clean-cpp
	find . -type d -name "__pycache__" | xargs rm -rf {};
	rm -rf $(VENV) .coverage .mypy_cache

.PHONY: lint
## Lint using flake8
lint:
	flake8 --ignore=W503,E501 ./src/

.PHONY: test
## Test using pytest
test:
	$(PYTHON) -m pytest

# .PHONY: ipynb2html
# ## Turn .ipynb file to html-with-toc
# ipynb2html:
#     $(VENV)/bin/jupyter nbconvert notebook_*.ipynb --template toc2 --CodeFoldingPreprocessor.remove_folded_code=True

##############################################################################
# Self Documenting Commands                                                  #
##############################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
