# CppGrad - lightweight Makefile build harness
#
# Targets:
#   make all                 - build + run tests and examples
#   make tests               - build and run the unit tests (what CI runs)
#   make tests-integration   - build and run heavy / checkpoint-gated tests (tests/integration/)
#   make tests-all           - tests + tests-integration
#   make examples            - build and run all examples
#   make build-metal         - build metal lib
#   make build-all           - build everything (no run)
#   make build-tests         - build test binaries only
#   make build-examples      - build example binaries only
#   make build-cmds          - build cmd binaries only
#   make run-tests           - run already-built tests
#   make run-examples        - run already-built examples
#   make clean               - remove build/

SHELL := /bin/sh

# ==== Platform detection ====
ON_APPLE := $(shell uname -s | grep -q Darwin && echo true || echo false)
HAS_XCRUN := $(shell command -v xcrun >/dev/null 2>&1 && echo true || echo false)
# Whether the Metal shader compiler actually runs -- not just that xcrun exists.
HAS_METAL := $(shell xcrun -sdk macosx metal --version >/dev/null 2>&1 && echo true || echo false)

# ==== Compiler and flags ====
CXX              ?= clang++
DEBUG            ?= false
SANITIZE_ADDRESS ?= false
SANITIZE_THREAD  ?= false
FFP_CONTRACT_OFF ?= false
FAST_MATH        ?= true
CXX_FLAGS        := -std=c++17

ifeq ($(DEBUG),true)
	CXX_FLAGS += -g -O0 -DCPPGRAD_DEBUG=1
else
	CXX_FLAGS += -O3
endif

ifeq ($(SANITIZE_ADDRESS),true)
	CXX_FLAGS += -fsanitize=address -fno-omit-frame-pointer
endif
ifeq ($(SANITIZE_THREAD),true)
	CXX_FLAGS += -fsanitize=thread
endif
ifeq ($(FFP_CONTRACT_OFF),true)
	CXX_FLAGS += -ffp-contract=off
endif
ifeq ($(FAST_MATH),false)
	CXX_FLAGS += -fno-fast-math
endif

INCLUDE_FLAGS := -Isrc -I.
OBJCXX_FLAGS  := -fobjc-arc

# ==== Homebrew headers/libs (macOS) ====
ifeq ($(ON_APPLE),true)
	FRAMEWORKS    := -framework Metal -framework Foundation -framework MetalPerformanceShaders
	INCLUDE_FLAGS += -I/opt/homebrew/include
	LIBRARY_FLAGS := -L/opt/homebrew/lib -lpcre2-8
	ifeq ($(HAS_METAL),true)
		CXX_FLAGS += -DCPPGRAD_WITH_METAL
	endif
else
	LIBRARY_FLAGS := -lpcre2-8
endif

# ==== Source discovery ====
LIB_CPP_SOURCES := $(shell find src -name '*.cpp' ! -path 'src/cppgrad/backend/metal/*')
# The Metal backend (.mm / .metal) is built only when the Metal compiler actually works; otherwise we
# build CPU-only (LIB_MM_SOURCES / LIB_METAL stay empty and CPPGRAD_WITH_METAL is undefined).
ifeq ($(HAS_METAL),true)
	LIB_MM_SOURCES := $(shell find src/cppgrad/backend/metal -name '*.mm' 2>/dev/null || true)
	LIB_METAL      := $(shell find src/cppgrad/backend/metal -name '*.metal' 2>/dev/null || true)
endif

# ==== Source paths ====
TEST_SRCS      := $(shell find tests -name '*.cpp' 2>/dev/null || true)
# Heavy / checkpoint-gated tests (large-model repros, speculative/MTP that need QWEN_MODEL_DIR)
# excluded from the default `tests` target (and CI) which runs the unit set;
TEST_INT_SRCS  := $(filter tests/integration/%,$(TEST_SRCS))
TEST_UNIT_SRCS := $(filter-out tests/integration/%,$(TEST_SRCS))
EXAMPLE_SRCS   := $(wildcard examples/*.cpp) $(shell find examples -mindepth 2 -name '*.cpp' 2>/dev/null || true)
# Standalone deliverable executables (e.g. the chat server). Unlike examples these ar build-only
# never auto-run (they block / need a model / take arguments). Each .cpp is a binary.
CMD_SRCS       := $(wildcard cmd/*.cpp) $(shell find cmd -mindepth 2 -name '*.cpp' 2>/dev/null || true)

# ==== Output paths ====
BUILD_DIR := build
METAL_DIR := $(BUILD_DIR)/metal
LIB_ARCHIVE := $(BUILD_DIR)/libcppgrad.a
METALLIB := $(METAL_DIR)/default.metallib

# Force-load static library so static initializers (device registration) run
ifeq ($(ON_APPLE),true)
	FORCE_LOAD := -Wl,-force_load,$(LIB_ARCHIVE)
else
	FORCE_LOAD := -Wl,--whole-archive $(LIB_ARCHIVE) -Wl,--no-whole-archive
endif

# Prevent Make from deleting .cpp.o / .mm.o (compound suffix → treated as intermediate)
.PRECIOUS: $(BUILD_DIR)/%.cpp.o $(BUILD_DIR)/%.mm.o

# Object files mirror source structure under build/
#   src/foo/bar.cpp        → build/src/foo/bar.cpp.o
#   tests/foo.cpp          → build/tests/foo.cpp.o
#   examples/sub/app.cpp   → build/examples/sub/app.cpp.o
LIB_CPP_OBJS := $(addprefix $(BUILD_DIR)/,$(LIB_CPP_SOURCES:.cpp=.cpp.o))
LIB_MM_OBJS  := $(addprefix $(BUILD_DIR)/,$(LIB_MM_SOURCES:.mm=.mm.o))

# ==== Binary paths ====
TEST_BINS     := $(patsubst %,$(BUILD_DIR)/%,$(TEST_UNIT_SRCS:.cpp=))
TEST_INT_BINS := $(patsubst %,$(BUILD_DIR)/%,$(TEST_INT_SRCS:.cpp=))
EXAMPLE_BINS  := $(patsubst %,$(BUILD_DIR)/%,$(EXAMPLE_SRCS:.cpp=))
CMD_BINS      := $(patsubst %,$(BUILD_DIR)/%,$(CMD_SRCS:.cpp=))

# ==== Linking flags per platform ====
LINK_PLATFORM_FLAGS := $(FRAMEWORKS)

# ==== Metal skipped message ====
ifeq ($(HAS_XCRUN),true)
define SKIPPED_METAL_MSG
@echo " Metal shader compiler missing - Metal Skipped (CPU-only build)."
@echo " To enable the Metal (GPU) backend, install the Metal toolchain:"
@echo "   xcodebuild -downloadComponent MetalToolchain"
endef
else
define SKIPPED_METAL_MSG
@echo "xcrun missing - Metal Skipped (CPU-only build)"
endef
endif

.PHONY: all tests tests-integration tests-all examples \
        build-metal build-all build-tests build-tests-integration build-examples build-cmds \
        run-tests run-tests-integration run-examples \
        clean

# ==== Default ====
all: build-metal tests examples

# ================================================================
# Metal - compile .metal → .air → .metallib (once, not per binary)
# Real file targets so make rebuilds incrementally: an .air recompiles only when its .metal changes,
# and the metallib relinks only when an .air changes.
# ================================================================
LIB_AIR := $(patsubst src/cppgrad/backend/metal/%.metal,$(METAL_DIR)/%.air,$(LIB_METAL))

$(METAL_DIR)/%.air: src/cppgrad/backend/metal/%.metal
	@mkdir -p $(METAL_DIR)
	@echo "Compiling Metal: $< -> $@"
	xcrun -sdk macosx metal -std=metal3.1 -O3 -c $< -o $@

$(METALLIB): $(LIB_AIR)
	@echo "Linking metallib: $@"
	xcrun -sdk macosx metallib $^ -o $@

ifeq ($(HAS_METAL),true)
build-metal: $(METALLIB)
else
build-metal:
	@echo "================================================================"
	$(SKIPPED_METAL_MSG)
	@echo "================================================================"
endif

# Binaries load `default.metallib` via [device newDefaultLibrary] -- i.e. from next to the executable
# -- so each binary depends on the metallib (see LINK_RULE) and the link step copies it alongside.
# A .metal edit thus relinks/recopies binaries but never recompiles C++ objects (they don't use it).
METALLIB_DEP := $(if $(filter true,$(HAS_METAL)),$(METALLIB),)

# ================================================================
# Static library - compile all library sources once into .a
# ================================================================
$(LIB_ARCHIVE): $(LIB_CPP_OBJS) $(LIB_MM_OBJS)
	ar rcs $@ $^

# ================================================================
# Single pattern rule - compiles any source to build/<src-path>.o
# ================================================================
$(BUILD_DIR)/%.cpp.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXX_FLAGS) $(INCLUDE_FLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/%.mm.o: %.mm
	@mkdir -p $(dir $@)
	$(CXX) $(CXX_FLAGS) $(INCLUDE_FLAGS) $(OBJCXX_FLAGS) -MMD -MP -c $< -o $@

# Header-dependency tracking: -MMD writes a .d file of header deps next to each .o, so editing a
# header recompiles every .cpp/.mm that includes it (prevents silent stale builds).
-include $(LIB_CPP_OBJS:.o=.d) $(LIB_MM_OBJS:.o=.d)
-include $(addprefix $(BUILD_DIR)/,$(TEST_SRCS:.cpp=.cpp.d) $(EXAMPLE_SRCS:.cpp=.cpp.d) $(CMD_SRCS:.cpp=.cpp.d))

# ================================================================
# Linking - one pattern rule handles both top-level and subdirs
# ================================================================
define LINK_RULE
$(BUILD_DIR)/$(1)/%: $(BUILD_DIR)/$(1)/%.cpp.o $(LIB_ARCHIVE) $(METALLIB_DEP)
	@mkdir -p $$(dir $$@)
	$$(CXX) $$(CXX_FLAGS) $$< $$(FORCE_LOAD) $$(LINK_PLATFORM_FLAGS) $$(LIBRARY_FLAGS) -o $$@
	@if [ "$(HAS_METAL)" = "true" ] && [ -f "$(METALLIB)" ]; then cp "$(METALLIB)" "$$(dir $$@)"; fi
endef

$(eval $(call LINK_RULE,tests))
$(eval $(call LINK_RULE,examples))
$(eval $(call LINK_RULE,cmd))

# ================================================================
# Build-only targets
# ================================================================
build-tests: $(TEST_BINS)

build-tests-integration: $(TEST_INT_BINS)

build-examples: $(EXAMPLE_BINS)

build-cmds: $(CMD_BINS)

build-all: build-metal build-tests build-tests-integration build-examples build-cmds

# ================================================================
# Run-only targets
# ================================================================
run-tests:
	@for bin in $(TEST_BINS); do \
		echo ""; \
		echo "======================================"; \
		echo "Running test: $$bin"; \
		echo "======================================"; \
		./"$$bin"; \
	done

run-tests-integration:
	@for bin in $(TEST_INT_BINS); do \
		echo ""; \
		echo "======================================"; \
		echo "Running integration test: $$bin"; \
		echo "======================================"; \
		./"$$bin"; \
	done

run-examples:
	@for bin in $(EXAMPLE_BINS); do \
		echo ""; \
		echo "======================================"; \
		echo "Running example: $$bin"; \
		echo "======================================"; \
		./"$$bin"; \
	done

# ================================================================
# Convenience - build then run
# ================================================================
# Ensure compilation finishes before running (with shared job tokens)
tests: build-tests
	+$(MAKE) run-tests

tests-integration: build-tests-integration
	+$(MAKE) run-tests-integration

tests-all: tests tests-integration

examples: build-examples
	+$(MAKE) run-examples

# ================================================================
# Clean
# ================================================================
clean:
	rm -rf $(BUILD_DIR)
