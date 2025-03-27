all: build

.PHONY: config-debug
config-debug:
	cmake -S . -B ./build -DCMAKE_BUILD_TYPE=Debug \
	-DCMAKE_TOOLCHAIN_FILE=${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake \
	-DCMAKE_PREFIX_PATH=${ORTOOLS_ROOT}

.PHONY: config-release
config-release:
	cmake -S . -B ./build -DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_TOOLCHAIN_FILE=${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake \
	-DCMAKE_PREFIX_PATH=${ORTOOLS_ROOT}

.PHONY: config
config: config-release

.PHONY: build
build:
	cmake --build ./build -j

.PHONY: build-sequential
build-sequential:
	cmake --build ./build

.PHONY: build-verbose
build-verbose:
	cmake --build ./build --verbose

.PHONY: clean
clean:
	rm -rf ./build

.PHONY: run
run:
	./build/playground/helloWorld

.PHONY: run-verbose
run-verbose:
	cp config_verbose.json config.json
	./build/userApplications/tiledCholesky
	
.PHONY: run-plain
run-plain:
	cp config_plain.json config.json
	./build/userApplications/tiledCholesky
	
.PHONY: run-lu
run-lu:
	./build/userApplications/lu_def --configFile=config_lu.json

.PHONY: run-lu-simple
run-lu-simple:
	./build/userApplications/lu_def --configFile=config_lu_simple.json

.PHONY: run-lu-verbose
run-lu-verbose:
	./build/userApplications/lu_def --configFile=config_lu.json --verbose
