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

.PHONY: clean pyclean run server ei_models_scan ei_res_unpack ei_db_scan ei_registry ei_figures_repack

pyclean:
	@find src tools -name '*.pyc' -exec rm -f {} +
	@find src tools -name '*.pyo' -exec rm -f {} +
	@find src tools -name __pycache__ -delete
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
	@PYTHONPATH=src $(PYTHON) -m client.main

# make server HOST=0.0.0.0 to accept connections from other devices
HOST ?= 127.0.0.1
PORT ?= 8280
server: venv_server
	@PYTHONPATH=src $(PYTHON_SERVER) -m server.main --host $(HOST) --port $(PORT)

# names of the figures and their textures of the maps of the game: figures_names.json and textures.json
ei_models_scan:
	@if [ ! -d "./repack/EIrepack" ]; then git clone --depth=1 https://github.com/aspadm/EIrepack.git ./repack/EIrepack; fi
	@cp -v tools/scan_models.py repack/EIrepack/
	@cd repack/EIrepack/ && PYTHONPATH="${PYTHONPATH}:Formats:." ../../venv_tools/bin/python3 scan_models.py ${SRC_DIR}/Maps/
# 	@mv -v repack/EIrepack/figures_names.json catalog/
# 	@mv -v repack/EIrepack/textures.json catalog/

# unpacks all archives of the game into UNPACK_DIR, writes figures_parts.json, figures_samples.json, buildings.json
# and plants.json of the maps, figures.json of the catalog is made by "make ei_registry" for every store
ei_res_unpack:
	@if [ ! -d "./repack/EIrepack" ]; then git clone --depth=1 https://github.com/aspadm/EIrepack.git ./repack/EIrepack; fi
	@cp -v tools/unpack_all.py repack/EIrepack/Converters/
	@cp -v tools/convert_model.py repack/EIrepack/Converters/
	@cd repack/EIrepack/ && PYTHONPATH="${PYTHONPATH}:Formats:." ../../venv_tools/bin/python3 Converters/unpack_all.py ${SRC_DIR} ${UNPACK_DIR} --verbose
# 	@mv -v repack/EIrepack/figures_parts.json catalog/
# 	@mv -v repack/EIrepack/figures_samples.json catalog/
# 	@mv -v repack/EIrepack/buildings.json catalog/
# 	@mv -v repack/EIrepack/plants.json catalog/

# tables of the database unpacked by "make ei_res_unpack" as JSON files in repack/EIrepack/ to look at them,
# the catalog is made of the database by "make ei_registry"
ei_db_scan:
	@if [ ! -d "./repack/EIrepack" ]; then git clone --depth=1 https://github.com/aspadm/EIrepack.git ./repack/EIrepack; fi
	@cp -v tools/extract_db_data.py repack/EIrepack/
	@cd repack/EIrepack/ && PYTHONPATH="${PYTHONPATH}:Formats:Converters:." ../../venv_tools/bin/python3 extract_db_data.py ${UNPACK_DIR}/Res/database/

# catalog of the units of the game for catalog/: races, monsters, NPCs, items, spells, animations and sounds of the figures,
# units of the maps, and figures.json, animations.json, armors.json, weapons.json, materials.json of the server and the client.
# The archives are read right from the game folders, nothing is unpacked. Every version of the game is a store, see lib.models:
# EI1_DIR is store 1 "Evil Islands" and EI2_DIR is store 2 "Lost in Astral", every file of the catalog has both of them.
# EI1_MOD_DIRS and EI2_MOD_DIRS are the folders of the mods which go over the game, for example
# EI2_DIR=.../ei2rebirth_unpacked EI2_MOD_DIRS=.../ei2rebirth_unpacked/Starter/Mods/LostInAstral
EI1_DIR ?= /run/media/veselin/STUFF/evil_islands/ei-1
EI2_DIR ?= /run/media/veselin/STUFF/evil_islands/ei-2
REGISTRY_DIR ?= catalog
ei_registry:
	@if [ ! -d "./repack/EIrepack" ]; then git clone --depth=1 https://github.com/aspadm/EIrepack.git ./repack/EIrepack; fi
	@cp -v tools/extract_db_data.py tools/build_registry.py repack/EIrepack/
	@cd repack/EIrepack/ && PYTHONPATH="${PYTHONPATH}:Formats:." ../../venv_tools/bin/python3 build_registry.py --store 1 ${EI1_DIR} ${EI1_MOD_DIRS} --store 2 ${EI2_DIR} ${EI2_MOD_DIRS} --dest $(CURDIR)/${REGISTRY_DIR}/

FIGURES_RES ?= $(HOME)/.worlds/data/figures.res
FIGURES_OUT ?= ./figures/
# split big figures.res into one .res file per model, those are downloaded by the client one by one
ei_figures_repack: venv
	@PYTHONPATH=src $(PYTHON) tools/split_figures.py $(FIGURES_RES) $(FIGURES_OUT)

tiles_build:
	@rm -rf ${RES_DIR}/merged_hq/*
	@PYTHONPATH=src samples_dir="${RES_DIR}/samples_hq/" venv_tools/bin/python3 tools/tiles.py merge_tiles ${RES_DIR}/shapes_hq/ ${RES_DIR}/groups_hq/ ${RES_DIR}/merged_hq/ ${RES_DIR}/ready_hq/
	@PYTHONPATH=src venv_tools/bin/python3 tools/tiles.py pack_tiles ${RES_DIR}/ready_hq/ ./assets/land/

fmg_clean:
	@rm -rf fmg/Fantasy-Map-Generator

fmg_build:
	@if [ ! -d "./fmg/Fantasy-Map-Generator" ]; then git clone https://github.com/vesellov/Fantasy-Map-Generator.git ./fmg/Fantasy-Map-Generator && cd ./fmg/Fantasy-Map-Generator && git switch --detach faf8a7f498d68427bdc66cbaf8abff53337c807e; fi
	@mkdir -p fmg/Fantasy-Map-Generator/tests/custom/
	@cp -v tools/options.js fmg/Fantasy-Map-Generator/public/modules/ui/
	@cp -v tools/burgs-generator.ts fmg/Fantasy-Map-Generator/src/modules/
	@cp -v tools/routes-generator.ts fmg/Fantasy-Map-Generator/src/modules/
	@cp -v tools/heightmap-templates.js fmg/Fantasy-Map-Generator/public/config/ 
	@cp -v tools/generate.spec.ts fmg/Fantasy-Map-Generator/tests/custom/
	@cp -v tools/playwright.config.custom.ts fmg/Fantasy-Map-Generator/
	@cd fmg/Fantasy-Map-Generator/ && npm install && npm install -D playwright@^1.64.1 @playwright/test@^1.64.1 && npm run build

fmg_generate:
	@cd fmg/Fantasy-Map-Generator/ && npx playwright test --config playwright.config.custom.ts
	@mv -v fmg/Fantasy-Map-Generator/map.json assets/
	@cat assets/map.json | python3 -m json.tool > assets/map_formatted.json

land_build:
	@rm -rfv assets/tiles.png
	@PYTHONPATH=src venv_tools/bin/python -m server.land assets/map.json 128 128
	@open assets/minimap.png
