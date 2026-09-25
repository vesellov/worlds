# This Makefile requires the following commands to be available:
# * java
# * virtualenv
# * python3.14
# * curl
# * apksigner
# * zipalign
# * apktool
# * Cython<3.0.0 (will be installed via pip)
#
# To install those tools you can use "system_dependencies" make target
#
# To be able to publish on Android Play Market need to first prepare the keystore file and send it Google:
# java -jar pepk.jar --keystore=~/keystores/app.keystore --alias=app --output=output.zip --encryptionkey=xxx --include-cert
# create a file ".keystore_password" in the same folder and store the keystore password there

ifeq ($(PYTHON),)
	PYTHON=python3.14
endif

REQUIREMENTS_ANDROID="python3,Kivy==2.3.0,android,numpy,pillow"

.DEFAULT_GOAL := build

.PHONY: clean


build: spec aab

.cache:
	@mkdir -p -v cache
	@if [ ! -f "./cache/pepk.jar" ]; then curl https://www.gstatic.com/play-apps-publisher-rapid/signing-tool/prod/pepk-src.jar -o ./cache/pepk.jar; fi
	@if [ ! -d "./cache/python-for-android" ]; then git clone --depth=1 https://github.com/kivy/python-for-android.git ./cache/python-for-android; fi
	@cd ./cache && git rev-list --count HEAD >revnum

.workspace:
	@rm -rf ./bin/* && rm -rf workspace && mkdir -p -v workspace
	@cp -r ./src/* workspace/ && cp -v ./cache/pepk.jar ./workspace/

venv:
	@$(PYTHON) -m venv venv
	@./venv/bin/pip install --upgrade pip setuptools
	@./venv/bin/pip install git+https://github.com/kivy/buildozer
	@./venv/bin/pip install legacy-cgi cython==0.29.34

.prepare: .cache .workspace
	@echo "current version is:" && cat ./workspace/version && echo ""
	@echo "core version is:" && cat ./workspace/core_version && echo ""
	@REVNUM=`cat ./cache/revnum`
	@echo "revision number based on total amount of commits in the Git repository is:" && cat ./cache/revnum
	@cd ./workspace/ && python3 -c "cv=open('core_version').read().strip().split('.');v=list(open('version').read().strip().split('.'));v[0]=cv[0];v[-2]=str(int(v[-2])+1);v[-1]=open('../cache/revnum').read().strip();open('version','w').write(('.'.join(v)).strip())"
	@echo "new version will be:" && cat ./workspace/version && echo ""
	@rm -rf ./workspace/app && mkdir -p -v ./workspace/app && cp -r ../../src/* ./workspace/app/
	@python3 -c "v=open('./workspace/version').read().strip();open('./workspace/app/version.py','w').write('__version__ = \"%s\"' % v);"

spec: venv .prepare
	@cd ./workspace/ && P_requirements="$(REQUIREMENTS_ANDROID)" $(PYTHON) -c "tpl=open('buildozer.spec.template').read();import os,sys;sys.stdout.write(tpl.format(requirements=os.environ['P_requirements']));" > buildozer.spec

apk: build_arm64_v8a build_x86_64

aab: .gradle_stop
	@rm -rfv ./workspace/bin/*.aab
	@cd ./workspace/ && PYTHONIOENCODING=utf-8 VIRTUAL_ENV=1 ../venv/bin/buildozer -v android release 1>buildozer.log 2>&1
	@cp -f -T -v ./workspace/bin/eiworlds*.aab ./workspace/bin/EIWorldsAndroid_unsigned.aab
	@cd ./workspace/ && zipalign -v 4 ./bin/EIWorldsAndroid_unsigned.aab ./bin/EIWorldsAndroid.aab 1>zipalign.log 2>&1
	@cd ./workspace/ && apksigner sign --ks ~/keystores/bitdust.keystore --ks-pass file:../.keystore_password --v1-signing-enabled true --v2-signing-enabled true --min-sdk-version 21 bin/EIWorldsAndroid.aab 1>apksigner.log 2>&1
	@rm -rfv ./workspace/bin/EIWorldsAndroid_unsigned.aab
	@rm -rfv ./workspace/bin/eiworlds-*-release.aab
	@mkdir -p ./bin/ && mv -v ./workspace/bin/*.aab ./bin/
	@cp -v ./workspace/version ./src/version && cat ./src/version && echo "" && echo "DONE"
	@echo "Now go to Goggle Dev Console at https://play.google.com/apps/publish/"
	@echo "Go to Releases -> Production page and click [Create new release] button"
	@echo "Upload bin/EIWorldsAndroid.aab file and copy release notes from the previos releases, then click [send release for review] button"

.gradle_stop:
	@(cd ~/.gradle/wrapper/dists/gradle-*-all/*/gradle-*/bin/ && ./gradle --stop) || echo "gradle daemon was not stopped"

debug: .gradle_stop spec
	@rm -rfv ./workspace/bin/*.apk
	@cd ./workspace/ && GRADLE_OPTS="-Xms1724m -Xmx5048m -Dorg.gradle.jvmargs='-Xms1724m -Xmx5048m'" PYTHONIOENCODING=utf-8 VIRTUAL_ENV=1 ../venv/bin/buildozer -v --profile arm64_v8a android debug
	@cp -v -T -f ./workspace/bin/eiworlds*.apk ./workspace/bin/EIWorldsAndroid_unsigned.apk
	@cd ./workspace/ && zipalign -v 4 ./bin/EIWorldsAndroid_unsigned.apk ./bin/EIWorldsAndroid.apk 1>zipalign.log 2>&1
	@cd ./workspace/ && apksigner sign --ks ~/keystores/bitdust.keystore --ks-pass file:../.keystore_password --v1-signing-enabled true --v2-signing-enabled true ./bin/EIWorldsAndroid.apk 1>apksigner.log 2>&1
	@mv -v ./workspace/bin/EIWorldsAndroid.apk ./workspace/bin/EIWorldsAndroid_arm64_v8a.apk
	@cp -v ./workspace/bin/EIWorldsAndroid_arm64_v8a.apk ./bin/
	@apktool d -o ./workspace/bin/apk/ -f ./workspace/bin/EIWorldsAndroid_arm64_v8a.apk 1>workspace/apktool.arm64_v8a.log 2>&1
	@mkdir -p -v ./workspace/bin/apk/private_mp3/
	@cp -v ./workspace/bin/apk/assets/private.tar ./workspace/bin/apk/private_mp3/private_mp3.tar.gz
	@cd ./workspace/bin/apk/private_mp3/ && tar -p -xf private_mp3.tar.gz && find . -type d -exec chmod +x {} \;
	@rm -rfv ./workspace/bin/EIWorldsAndroid.apk
	@rm -rfv ./workspace/bin/EIWorldsAndroid_unsigned.apk
	@rm -rfv ./workspace/bin/eiworlds-*-release-unsigned.apk

build_arm64_v8a: .gradle_stop
	@cd ./workspace/ && PYTHONIOENCODING=utf-8 VIRTUAL_ENV=1 ../venv/bin/buildozer -v --profile arm64_v8a android release 1>buildozer.arm64_v8a.log 2>&1
	@cp -v -T -f ./workspace/bin/eiworlds*.apk ./workspace/bin/EIWorldsAndroid_unsigned.apk
	@cd ./workspace/ && zipalign -v 4 ./bin/EIWorldsAndroid_unsigned.apk ./bin/EIWorldsAndroid.apk 1>zipalign.arm64_v8a.log 2>&1
	@cd ./workspace/ && apksigner sign --ks ~/keystores/bitdust.keystore --ks-pass file:../.keystore_password --v1-signing-enabled true --v2-signing-enabled true ./bin/EIWorldsAndroid.apk 1>apksigner.arm64_v8a.log 2>&1
	@mv -v ./workspace/bin/EIWorldsAndroid.apk ./workspace/bin/EIWorldsAndroid_arm64_v8a.apk
	@apktool d -o ./workspace/bin/apk/ -f ./workspace/bin/EIWorldsAndroid_arm64_v8a.apk 1>workspace/apktool.arm64_v8a.log 2>&1
	@mkdir -p ./bin/ && cp -v ./workspace/bin/EIWorldsAndroid_arm64_v8a.apk ./bin/
	@cp -v ./workspace/version ./src/version && cat ./src/version && echo "" && echo "DONE"

build_x86_64: .gradle_stop
	@cd ./workspace/ && PYTHONIOENCODING=utf-8 VIRTUAL_ENV=1 ../venv/bin/buildozer -v --profile x86_64 android release 1>buildozer.x86_64.log 2>&1
	@cp -v -T -f ./workspace/bin/eiworlds*.apk ./workspace/bin/EIWorldsAndroid_unsigned.apk
	@cd ./workspace/ && zipalign -v 4 ./bin/EIWorldsAndroid_unsigned.apk  ./bin/EIWorldsAndroid.apk 1>zipalign.x86_64.log 2>&1
	@cd ./workspace/ && apksigner sign --ks ~/keystores/bitdust.keystore --ks-pass file:../.keystore_password --v1-signing-enabled true --v2-signing-enabled true ./bin/EIWorldsAndroid.apk 1>apksigner.x86_64.log 2>&1
	@mv -v ./workspace/bin/EIWorldsAndroid.apk ./workspace/bin/EIWorldsAndroid_x86_64.apk
	@apktool d -o ./workspace/bin/apk/ -f ./workspace/bin/EIWorldsAndroid_x86_64.apk 1>workspace/apktool.x86_64.log 2>&1
	@mkdir -p ./bin/ && cp -v ./workspace/bin/EIWorldsAndroid_x86_64.apk ./bin/
	@cp -v ./workspace/version ./src/version && cat ./src/version && echo "" && echo "DONE"

clean:
	@(cd ./workspace/ && PYTHONIOENCODING=utf-8 VIRTUAL_ENV=1 ../venv/bin/buildozer -v android clean) || echo "done"

full_clean:
	@rm -rf ./venv
	@rm -rf ./workspace
	@rm -rf ./cache/.buildozer
	@rm -rf ./cache/*
	@rm -rf ./bin/*

install_system_packages:
	@sudo apt update; sudo apt install --yes --no-install-recommends openjdk-17-jdk cython3 autoconf automake libtool ccache lld zipalign apksigner apktool adb

test:
	@adb install -r -d bin/EIWorldsAndroid_arm64_v8a.apk

shell:
	@echo "run-as org.eiworlds.eiworlds"
	@adb shell "run-as org.eiworlds.eiworlds; bash"

log:
	@cd ./workspace/ && PYTHONIOENCODING=utf-8 VIRTUAL_ENV=1 ../venv/bin/buildozer -v android logcat

log_full:
	@adb logcat *:V python:V

log_quick:
	@adb logcat | grep -E "python|PythonActivity|SDL|PythonService|crush|Exception"
