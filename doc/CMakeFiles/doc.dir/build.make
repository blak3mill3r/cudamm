# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.6

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
CMAKE_SOURCE_DIR = /home/blake/w/cudamm

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/blake/w/cudamm

# Utility rule file for doc.

doc/CMakeFiles/doc:
	cd /home/blake/w/cudamm/doc && /usr/bin/doxygen

doc: doc/CMakeFiles/doc
doc: doc/CMakeFiles/doc.dir/build.make
.PHONY : doc

# Rule to build all files generated by this target.
doc/CMakeFiles/doc.dir/build: doc
.PHONY : doc/CMakeFiles/doc.dir/build

doc/CMakeFiles/doc.dir/clean:
	cd /home/blake/w/cudamm/doc && $(CMAKE_COMMAND) -P CMakeFiles/doc.dir/cmake_clean.cmake
.PHONY : doc/CMakeFiles/doc.dir/clean

doc/CMakeFiles/doc.dir/depend:
	cd /home/blake/w/cudamm && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/blake/w/cudamm /home/blake/w/cudamm/doc /home/blake/w/cudamm /home/blake/w/cudamm/doc /home/blake/w/cudamm/doc/CMakeFiles/doc.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : doc/CMakeFiles/doc.dir/depend

