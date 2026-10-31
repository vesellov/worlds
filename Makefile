# This Makefile requires the following commands to be available:
# * python3
# * virtualenv

ifeq ($(PYTHON_VERSION),)
	PYTHON_VERSION=python3.12
endif

REQUIREMENTS_TXT:=requirements.txt
REQUIREMENTS_TOOLS_TXT:=requirements_tools.txt
REQUIREMENTS_SERVER_TXT:=requirements_server.txt
OS=$(shell lsb_release -si 2>/dev/null || uname)
PIP:="venv/bin/pip3"
PYTHON="venv/bin/python3"
PIP_TOOLS:="venv_tools/bin/pip3"
PYTHON_TOOLS="venv_tools/bin/python3"
PIP_SERVER:="venv_server/bin/pip3"
PYTHON_SERVER="venv_server/bin/python3"

.DEFAULT_GOAL := venv

.PHONY: clean pyclean run server

pyclean:
	@find src -name '*.pyc' -exec rm -f {} +
	@find src -name '*.pyo' -exec rm -f {} +
	@find src -name __pycache__ -delete
	@find . -name .DS_Store -delete
	@rm -rf *.egg-info build
	@rm -rf coverage.xml .coverage

clean: pyclean
	@rm -rf venv

venv:
	@$(PYTHON_VERSION) -m venv venv
	@$(PIP) install --upgrade pip setuptools wheel cython
	@$(PIP) install kivy[base]==2.3.1 --no-cache-dir --no-binary kivy
	@$(PIP) install -r $(REQUIREMENTS_TXT) --no-cache-dir
	@# PIP_NO_BINARY=kivy $(PIP) install -r $(REQUIREMENTS_TXT)

venv_server:
	@$(PYTHON_VERSION) -m venv venv_server
	@$(PIP_SERVER) install --upgrade pip
	@$(PIP_SERVER) install -r $(REQUIREMENTS_SERVER_TXT)

venv_tools:
	@$(PYTHON_VERSION) -m venv venv_tools
	@$(PIP_TOOLS) install --upgrade pip
	@$(PIP_TOOLS) install -r $(REQUIREMENTS_TOOLS_TXT)

run: venv
	@$(PYTHON) src/main.py

# make server HOST=0.0.0.0 to accept connections from other devices
HOST ?= 127.0.0.1
PORT ?= 8280
server: venv_server
	@$(PYTHON_SERVER) src/server/main.py --host $(HOST) --port $(PORT)

ei_models_scan:
	@if [ ! -d "./repack/EIrepack" ]; then git clone --depth=1 https://github.com/aspadm/EIrepack.git ./repack/EIrepack; fi
	@cp -v tools/scan_models.py repack/EIrepack/
	@cd repack/EIrepack/ && PYTHONPATH="${PYTHONPATH}:Formats:." ../../venv_tools/bin/python3 scan_models.py ${SRC_DIR}/Maps/
	@mv -v repack/EIrepack/figures_names.json catalog/
	@mv -v repack/EIrepack/textures.json catalog/

ei_res_unpack:
	@if [ ! -d "./repack/EIrepack" ]; then git clone --depth=1 https://github.com/aspadm/EIrepack.git ./repack/EIrepack; fi
	@cp -v tools/unpack_all.py repack/EIrepack/Converters/
	@cp -v tools/convert_model.py repack/EIrepack/Converters/
	@cd repack/EIrepack/ && PYTHONPATH="${PYTHONPATH}:Formats:." ../../venv_tools/bin/python3 Converters/unpack_all.py ${SRC_DIR} ${UNPACK_DIR} --verbose
	@mv -v repack/EIrepack/figures.json catalog/
	@mv -v repack/EIrepack/figures_parts.json catalog/
	@mv -v repack/EIrepack/figures_samples.json catalog/
	@mv -v repack/EIrepack/buildings.json catalog/
	@mv -v repack/EIrepack/plants.json catalog/

ei_figures_scan:
	@if [ ! -d "./repack/EIrepack" ]; then git clone --depth=1 https://github.com/aspadm/EIrepack.git ./repack/EIrepack; fi
	@cp -v tools/scan_figures.py repack/EIrepack/
	@cp -v catalog/figures.json repack/EIrepack/
	@cd repack/EIrepack/ && PYTHONPATH="${PYTHONPATH}:Formats:Converters:." ../../venv_tools/bin/python3 scan_figures.py ${UNPACK_DIR}/Res/figures/
	@mv -v repack/EIrepack/figures.json catalog/

ei_db_scan:
	@if [ ! -d "./repack/EIrepack" ]; then git clone --depth=1 https://github.com/aspadm/EIrepack.git ./repack/EIrepack; fi
	@cp -v tools/extract_db_data.py repack/EIrepack/
	@cd repack/EIrepack/ && PYTHONPATH="${PYTHONPATH}:Formats:Converters:." ../../venv_tools/bin/python3 extract_db_data.py ${UNPACK_DIR}/Res/database/

ei_compile:
	@if [ ! -d "./repack/EIrepack" ]; then git clone --depth=1 https://github.com/aspadm/EIrepack.git ./repack/EIrepack; fi
	@cp -v tools/compile_data.py repack/EIrepack/
	@cd repack/EIrepack/ && PYTHONPATH="${PYTHONPATH}:Formats:Converters:." ../../venv_tools/bin/python3 compile_data.py
	@mv -v repack/EIrepack/armors.json catalog/
	@mv -v repack/EIrepack/weapons.json catalog/
	@mv -v repack/EIrepack/materials.json catalog/
	@mv -v repack/EIrepack/animations.json catalog/

FIGURES_RES ?= $(HOME)/.worlds/data/figures.res
FIGURES_OUT ?= ./figures/
ei_figures_repack: venv
	# split big figures.res into one .res file per model, those are downloaded by the client one by one
	@$(PYTHON) tools/split_figures_res.py $(FIGURES_RES) $(FIGURES_OUT)

tiles_build:
	@rm -rf ${RES_DIR}/merged_hq/*
	@samples_dir="${RES_DIR}/samples_hq/" venv_tools/bin/python3 src/tiles.py merge_tiles ${RES_DIR}/shapes_hq/ ${RES_DIR}/groups_hq/ ${RES_DIR}/merged_hq/ ${RES_DIR}/ready_hq/
	@venv_tools/bin/python3 src/tiles.py pack_tiles ${RES_DIR}/ready_hq/ ./assets/land/

fmg_clean:
	@rm -rf fmg/Fantasy-Map-Generator

fmg_build:
	@if [ ! -d "./fmg/Fantasy-Map-Generator" ]; then git clone --depth=1 https://github.com/Azgaar/Fantasy-Map-Generator.git ./fmg/Fantasy-Map-Generator; fi
	@mkdir -p fmg/Fantasy-Map-Generator/tests/custom/
	@cp -v tools/options.js fmg/Fantasy-Map-Generator/public/modules/ui/
	@cp -v tools/burgs-generator.ts fmg/Fantasy-Map-Generator/src/modules/
	@cp -v tools/routes-generator.ts fmg/Fantasy-Map-Generator/src/modules/
	@cp -v tools/heightmap-templates.js fmg/Fantasy-Map-Generator/public/config/ 
	@cp -v tools/generate.spec.ts fmg/Fantasy-Map-Generator/tests/custom/
	@cp -v tools/playwright.config.custom.ts fmg/Fantasy-Map-Generator/
	@cd fmg/Fantasy-Map-Generator/ && npm install && npx playwright install && npm run build

fmg_generate:
	@cd fmg/Fantasy-Map-Generator/ && npx playwright test --config playwright.config.custom.ts
	@mv -v fmg/Fantasy-Map-Generator/map.json assets/
	@cat assets/map.json | python3 -m json.tool > assets/map_formatted.json

land_build:
	@rm -rfv assets/tiles.png
	@venv_tools/bin/python src/land.py map.json 512 512
	@open assets/minimap.png
