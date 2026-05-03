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

fmg_clone:
	@if [ ! -d "./fmg/Fantasy-Map-Generator" ]; then git clone --depth=1 https://github.com/Azgaar/Fantasy-Map-Generator.git ./fmg/Fantasy-Map-Generator; fi

fmg_build:
	@mkdir -p fmg/Fantasy-Map-Generator/tests/custom/
	@cp -v tools/heightmap-templates.js fmg/Fantasy-Map-Generator/public/config/ 
	@cp -v tools/generate.spec.ts fmg/Fantasy-Map-Generator/tests/custom/
	@cp -v tools/playwright.config.custom.ts fmg/Fantasy-Map-Generator/
	@cd fmg/Fantasy-Map-Generator/ && npm install && npm run build

fmg_generate:
	@cd fmg/Fantasy-Map-Generator/ && npx playwright install && npx playwright test --config playwright.config.custom.ts
	@mv -v fmg/Fantasy-Map-Generator/map.json .
	@cat map.json | python3 -m json.tool > map_formatted.json

land_build:
	@rm -rfv assets/tiles.png
	@venv/bin/python src/land.py map.json 512 512
	@open assets/tiles.png
