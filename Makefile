# This Makefile requires the following commands to be available:
# * python3
# * virtualenv

ifeq ($(PYTHON_VERSION),)
	PYTHON_VERSION=python3
endif

REQUIREMENTS_TXT:=requirements.txt
OS=$(shell lsb_release -si 2>/dev/null || uname)
PIP:="venv/bin/pip3"
PYTHON="venv/bin/python3"

.DEFAULT_GOAL := venv

.PHONY: clean pyclean

pyclean:
	@find . -name '*.pyc' -exec rm -f {} +
	@find . -name '*.pyo' -exec rm -f {} +
	@find . -name __pycache__ -delete
	@find . -name .DS_Store -delete
	@rm -rf *.egg-info build
	@rm -rf coverage.xml .coverage

clean: pyclean
	@rm -rf venv

venv:
	@$(PYTHON_VERSION) -m venv venv
	@$(PIP) install --upgrade pip
	@$(PIP) install -r $(REQUIREMENTS_TXT)

run: venv
	@$(PYTHON) src/main.py

res_unpack:
	@if [ ! -d "./repack/EIrepack" ]; then git clone --depth=1 https://github.com/aspadm/EIrepack.git ./repack/EIrepack; fi
	@cp -v tools/scan_models.py repack/EIrepack/
	@cp -v tools/scan_figures.py repack/EIrepack/
	@cp -v tools/unpack_all.py repack/EIrepack/Converters/
	@cp -v tools/convert_model.py repack/EIrepack/Converters/
	@cd repack/EIrepack/ && PYTHONPATH="${PYTHONPATH}:Formats:." ../../venv/bin/python3 scan_models.py ${SRC_DIR}/Maps/
	@cd repack/EIrepack/ && PYTHONPATH="${PYTHONPATH}:Formats:." ../../venv/bin/python3 Converters/unpack_all.py ${SRC_DIR} ${DEST_DIR} --verbose
	@cd repack/EIrepack/ && PYTHONPATH="${PYTHONPATH}:Formats:Converters:." ../../venv/bin/python3 scan_figures.py ${DEST_DIR}/Res/figures/
	@mv -v repack/EIrepack/templates.json assets/
	@mv -v repack/EIrepack/figures.json assets/
	@mv -v repack/EIrepack/catalog_buildings.json assets/
	@mv -v repack/EIrepack/catalog_plants.json assets/
	@mv -v repack/EIrepack/catalog_figures.json assets/

tiles_build:
	@rm -rf ${RES_DIR}/merged_hq/*
	@samples_dir="${RES_DIR}/samples_hq/" venv/bin/python3 src/tiles.py merge_tiles ${RES_DIR}/shapes_hq/ ${RES_DIR}/groups_hq/ ${RES_DIR}/merged_hq/ ${RES_DIR}/ready_hq/
	@venv/bin/python3 src/tiles.py pack_tiles ${RES_DIR}/ready_hq/ ./assets/land/

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
	@mv -v fmg/Fantasy-Map-Generator/map.json .
	@cat map.json | python3 -m json.tool > map_formatted.json

land_build:
	@rm -rfv assets/tiles.png
	@venv/bin/python src/land.py map.json 512 512
	@open assets/minimap.png
