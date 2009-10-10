# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.6

# Default target executed when no arguments are given to make.
default_target: all
.PHONY : default_target

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canoncical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/bin/ccmake

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/blake/src/cudamm

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/blake/src/cudamm

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake cache editor..."
	/usr/bin/ccmake -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache
.PHONY : edit_cache/fast

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/usr/bin/cmake -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache
.PHONY : rebuild_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /home/blake/src/cudamm/CMakeFiles /home/blake/src/cudamm/CMakeFiles/progress.make
	$(MAKE) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /home/blake/src/cudamm/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean
.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named cudamm

# Build rule for target.
cudamm: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 cudamm
.PHONY : cudamm

# fast build rule for target.
cudamm/fast:
	$(MAKE) -f src/CMakeFiles/cudamm.dir/build.make src/CMakeFiles/cudamm.dir/build
.PHONY : cudamm/fast

#=============================================================================
# Target rules for targets named pycudamm

# Build rule for target.
pycudamm: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 pycudamm
.PHONY : pycudamm

# fast build rule for target.
pycudamm/fast:
	$(MAKE) -f python/CMakeFiles/pycudamm.dir/build.make python/CMakeFiles/pycudamm.dir/build
.PHONY : pycudamm/fast

#=============================================================================
# Target rules for targets named cudamm-test

# Build rule for target.
cudamm-test: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 cudamm-test
.PHONY : cudamm-test

# fast build rule for target.
cudamm-test/fast:
	$(MAKE) -f test/CMakeFiles/cudamm-test.dir/build.make test/CMakeFiles/cudamm-test.dir/build
.PHONY : cudamm-test/fast

#=============================================================================
# Target rules for targets named test.cubin

# Build rule for target.
test.cubin: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 test.cubin
.PHONY : test.cubin

# fast build rule for target.
test.cubin/fast:
	$(MAKE) -f test/CMakeFiles/test.cubin.dir/build.make test/CMakeFiles/test.cubin.dir/build
.PHONY : test.cubin/fast

#=============================================================================
# Target rules for targets named doc

# Build rule for target.
doc: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 doc
.PHONY : doc

# fast build rule for target.
doc/fast:
	$(MAKE) -f doc/CMakeFiles/doc.dir/build.make doc/CMakeFiles/doc.dir/build
.PHONY : doc/fast

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... edit_cache"
	@echo "... rebuild_cache"
	@echo "... cudamm"
	@echo "... pycudamm"
	@echo "... cudamm-test"
	@echo "... test.cubin"
	@echo "... doc"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system
