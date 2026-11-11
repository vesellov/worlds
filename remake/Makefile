# Evil Islands remake on Python + Kivy, see README.md
# The folder of the original game is taken from EI_DATA_PATH.

ifeq ($(PYTHON_VERSION),)
	PYTHON_VERSION=python3.12
endif

PIP:=venv/bin/pip
PYTHON:=venv/bin/python
RUN:=PYTHONPATH=src $(PYTHON)

.DEFAULT_GOAL := venv

GHIDRA_HOME ?= ../opt/ghidra
GHIDRA_JAVA_HOME ?= /usr/lib/jvm/java-25-openjdk-amd64
GHIDRA_PROJECT ?= build/ghidra
EI_DATA_PATH ?= /run/media/veselin/STUFF/evil_islands/EvilIslands/ENG/game_info/data
EI2_DATA_PATH ?= /run/media/veselin/STUFF/evil_islands/EvilIslands2/game_info/data
GHIDRA2_PROJECT ?= build/ghidra2

.PHONY: venv clean pyclean test model map menu demo snapshot registry coverage ghidra ghidra2

venv:
	@$(PYTHON_VERSION) -m venv venv
	@$(PIP) install --upgrade pip setuptools wheel
	@$(PIP) install -r requirements.txt

pyclean:
	@find src tests -name __pycache__ -prune -exec rm -rf {} +

clean: pyclean
	@rm -rf venv

test:
	@$(RUN) -m unittest discover -s tests

model:
	@$(RUN) -m client.main model $(or $(NAME),unhuma)

map:
	@$(RUN) -m client.main map $(or $(NAME),bz10k)

menu:
	@$(RUN) -m client.main menu

# the game as game.exe starts it with everything the remake has: the main menu, "New game" -> zone1 with the hero,
# its weapons and spells (F1 in the game shows the keys)
demo:
	@$(RUN) -m client.main demo

snapshot:
	@$(RUN) -m tools.snapshot figure $(or $(NAME),unhuma) $(ACTION) -o snapshot.png

# the functions of build/ghidra/game_exe.c and the code of the remake which implements them:
# func_registry.json and FUNC_REGISTRY.md
registry:
	@$(RUN) -m tools.func_registry

# run game.exe under the coverage tracer re/eitrace with eitools/scenarios/coverage.txt, result in re/coverage
coverage:
	@$(RUN) -m tools.func_coverage run coverage

# import and analyze game.exe, apply re/names.json, write build/ghidra/game_exe.c
ghidra:
	@mkdir -p $(GHIDRA_PROJECT)
	@JAVA_HOME=$(GHIDRA_JAVA_HOME) $(GHIDRA_HOME)/support/analyzeHeadless $(GHIDRA_PROJECT) EvilIslands \
		-import $(EI_DATA_PATH)/game.exe -overwrite -scriptPath re -postScript ApplyNames.java \
		re/names.json $(abspath $(GHIDRA_PROJECT))

# the same for game.exe of the second version (2006) to compare them, without the names of the first one
ghidra2:
	@mkdir -p $(GHIDRA2_PROJECT)
	@JAVA_HOME=$(GHIDRA_JAVA_HOME) $(GHIDRA_HOME)/support/analyzeHeadless $(GHIDRA2_PROJECT) EvilIslands2 \
		-import $(EI2_DATA_PATH)/game.exe -overwrite -scriptPath re -postScript ApplyNames.java \
		- $(abspath $(GHIDRA2_PROJECT))
