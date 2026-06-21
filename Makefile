# CppGrad - lightweight Makefile build harness
#
# Targets:
#   make all                 - build + run tests and examples
#   make tests               - build and run all tests
#   make examples            - build and run all examples
#   make examples-except-llm - build and run examples (exclude heavy LLM)
#   make build-metal         - build metal lib
#   make build-all           - build everything (no run)
#   make build-tests         - build test binaries only
#   make build-examples      - build example binaries only
#   make run-tests           - run already-built tests
#   make run-examples        - run already-built examples
#   make clean               - remove build/
#
# The scripts/build_common.sh is kept for interactive use; this Makefile
# duplicates the flag logic so each binary compiles independently in parallel.

SHELL := /bin/sh

# ==== Platform detection ====
ON_APPLE := $(shell uname -s | grep -q Darwin && echo true || echo false)
HAS_XCRUN := $(shell command -v xcrun >/dev/null 2>&1 && echo true || echo false)

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

# Homebrew headers/libs (macOS)
ifeq ($(ON_APPLE),true)
	FRAMEWORKS    := -framework Metal -framework Foundation -framework MetalPerformanceShaders
	INCLUDE_FLAGS += -I/opt/homebrew/include
	LIBRARY_FLAGS := -L/opt/homebrew/lib -lpcre2-8
	ifeq ($(HAS_XCRUN),true)
		CXX_FLAGS += -DCPPGRAD_WITH_METAL
	endif
else
	LIBRARY_FLAGS := -lpcre2-8
endif

# ==== Source discovery ====
LIB_CPP_SOURCES := $(shell find src -name '*.cpp' ! -path 'src/cppgrad/backend/metal/*')
ifeq ($(ON_APPLE)$(HAS_XCRUN),truetrue)
	LIB_MM_SOURCES := $(shell find src/cppgrad/backend/metal -name '*.mm' 2>/dev/null || true)
	LIB_METAL      := $(shell find src/cppgrad/backend/metal -name '*.metal' 2>/dev/null || true)
endif

TEST_SRCS          := $(wildcard tests/*.cpp)
EXAMPLE_SRCS       := $(wildcard examples/*.cpp) $(shell find examples -mindepth 2 -name '*.cpp' 2>/dev/null || true)
EXAMPLE_LIGHT_SRCS := $(filter-out examples/llm/qwen3_inference.cpp,$(EXAMPLE_SRCS))

# ==== Output paths ====
BUILD_DIR := build
METAL_DIR := $(BUILD_DIR)/metal
LIB_ARCHIVE := $(BUILD_DIR)/libcppgrad.a

# Force-load static library so static initializers (device registration) run
ifeq ($(ON_APPLE),true)
	FORCE_LOAD := -Wl,-force_load,$(LIB_ARCHIVE)
else
	FORCE_LOAD := -Wl,--whole-archive $(LIB_ARCHIVE) -Wl,--no-whole-archive
endif

METALLIB    := $(METAL_DIR)/default.metallib
METAL_SKIP  := $(METAL_DIR)/.skip

# Prevent Make from deleting .cpp.o / .mm.o (compound suffix → treated as intermediate)
.PRECIOUS: $(BUILD_DIR)/%.cpp.o $(BUILD_DIR)/%.mm.o

# Object files mirror source structure under build/
#   src/foo/bar.cpp        → build/src/foo/bar.cpp.o
#   tests/foo.cpp          → build/tests/foo.cpp.o
#   examples/sub/app.cpp   → build/examples/sub/app.cpp.o
LIB_CPP_OBJS := $(addprefix $(BUILD_DIR)/,$(LIB_CPP_SOURCES:.cpp=.cpp.o))
LIB_MM_OBJS  := $(addprefix $(BUILD_DIR)/,$(LIB_MM_SOURCES:.mm=.mm.o))

# Binary paths
TEST_BINS          := $(patsubst %,$(BUILD_DIR)/%,$(TEST_SRCS:.cpp=))
EXAMPLE_BINS       := $(patsubst %,$(BUILD_DIR)/%,$(EXAMPLE_SRCS:.cpp=))
EXAMPLE_LIGHT_BINS := $(patsubst %,$(BUILD_DIR)/%,$(EXAMPLE_LIGHT_SRCS:.cpp=))

# ==== Linking flags per platform ====
LINK_PLATFORM_FLAGS := $(FRAMEWORKS)
ifeq ($(ON_APPLE),true)
	COPY_METALLIB = @test -f $(METALLIB) && cp $(METALLIB) "$(dir $@)" || true
else
	COPY_METALLIB =
endif

.PHONY: all tests examples examples-except-llm \
        build-metal build-all build-tests build-examples build-except-llm \
        run-tests run-examples run-except-llm \
        clean

# ==== Default ====
all: build-metal tests examples

# ================================================================
# Metal - compile .metal → .air → .metallib (once, not per binary)
# ================================================================
build-metal:
ifneq ($(HAS_XCRUN),true)
	@mkdir -p $(METAL_DIR)
	@test -f $(METAL_SKIP) || { echo "xcrun not found - Metal skipped"; touch $(METAL_SKIP); }
else
	@mkdir -p $(METAL_DIR)
	@rm -f $(METAL_SKIP)
	@AIRS=; \
	for m in $(LIB_METAL); do \
		base="$$(basename "$$m" .metal)"; \
		if [ ! -f "$(METAL_DIR)/$$base.air" ] || [ "$$m" -nt "$(METAL_DIR)/$$base.air" ]; then \
			echo "Compiling Metal: $$m -> $(METAL_DIR)/$$base.air"; \
			xcrun -sdk macosx metal -std=metal3.1 -O3 -c "$$m" -o "$(METAL_DIR)/$$base.air"; \
		fi; \
		AIRS="$$AIRS $(METAL_DIR)/$$base.air"; \
	done; \
	if [ -n "$$AIRS" ]; then \
		echo "Linking metallib: $(METALLIB)"; \
		xcrun -sdk macosx metallib $$AIRS -o "$(METALLIB)"; \
	fi
endif

# Force all object files to wait until Metal shader compilation is complete
$(LIB_CPP_OBJS) $(LIB_MM_OBJS): build-metal

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
	$(CXX) $(CXX_FLAGS) $(INCLUDE_FLAGS) -c $< -o $@

$(BUILD_DIR)/%.mm.o: %.mm
	@mkdir -p $(dir $@)
	$(CXX) $(CXX_FLAGS) $(INCLUDE_FLAGS) $(OBJCXX_FLAGS) -c $< -o $@

# ================================================================
# Linking - one pattern rule handles both top-level and subdirs
# ================================================================
$(BUILD_DIR)/tests/%: $(BUILD_DIR)/tests/%.cpp.o $(LIB_ARCHIVE)
	@mkdir -p $(dir $@)
	$(CXX) $(CXX_FLAGS) $< $(FORCE_LOAD) $(LINK_PLATFORM_FLAGS) $(LIBRARY_FLAGS) -o $@
	$(COPY_METALLIB)

$(BUILD_DIR)/examples/%: $(BUILD_DIR)/examples/%.cpp.o $(LIB_ARCHIVE)
	@mkdir -p $(dir $@)
	$(CXX) $(CXX_FLAGS) $< $(FORCE_LOAD) $(LINK_PLATFORM_FLAGS) $(LIBRARY_FLAGS) -o $@
	$(COPY_METALLIB)

# ================================================================
# Build-only targets
# ================================================================
build-tests: $(TEST_BINS)

build-examples: $(EXAMPLE_BINS)

build-except-llm: $(EXAMPLE_LIGHT_BINS)

build-all: build-metal build-tests build-examples

# ================================================================
# Run-only targets
# ================================================================
run-tests:
	@for bin in $(TEST_BINS); do \
		echo ""; \
		echo "======================================"; \
		echo "Running test: $$bin"; \
		echo "======================================"; \
		"$$bin"; \
	done

run-examples:
	@for bin in $(EXAMPLE_BINS); do \
		echo ""; \
		echo "======================================"; \
		echo "Running example: $$bin"; \
		echo "======================================"; \
		"$$bin"; \
	done

run-except-llm:
	@for bin in $(EXAMPLE_LIGHT_BINS); do \
		echo ""; \
		echo "======================================"; \
		echo "Running example: $$bin"; \
		echo "======================================"; \
		"$$bin"; \
	done

# ================================================================
# Convenience - build then run
# ================================================================
# Ensure compilation finishes before running (with shared job tokens)
tests: build-tests
	+$(MAKE) run-tests

examples: build-examples
	+$(MAKE) run-examples

examples-except-llm: build-except-llm
	+$(MAKE) run-except-llm

# ================================================================
# Clean
# ================================================================
clean:
	rm -rf $(BUILD_DIR)
