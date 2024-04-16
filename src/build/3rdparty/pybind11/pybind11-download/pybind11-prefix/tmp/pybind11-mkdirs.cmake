# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "H:/shliao/DCVC/DCVC-DC/src/build/3rdparty/pybind11/pybind11-src"
  "H:/shliao/DCVC/DCVC-DC/src/build/3rdparty/pybind11/pybind11-build"
  "H:/shliao/DCVC/DCVC-DC/src/build/3rdparty/pybind11/pybind11-download/pybind11-prefix"
  "H:/shliao/DCVC/DCVC-DC/src/build/3rdparty/pybind11/pybind11-download/pybind11-prefix/tmp"
  "H:/shliao/DCVC/DCVC-DC/src/build/3rdparty/pybind11/pybind11-download/pybind11-prefix/src/pybind11-stamp"
  "H:/shliao/DCVC/DCVC-DC/src/build/3rdparty/pybind11/pybind11-download/pybind11-prefix/src"
  "H:/shliao/DCVC/DCVC-DC/src/build/3rdparty/pybind11/pybind11-download/pybind11-prefix/src/pybind11-stamp"
)

set(configSubDirs Debug;Release;MinSizeRel;RelWithDebInfo)
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "H:/shliao/DCVC/DCVC-DC/src/build/3rdparty/pybind11/pybind11-download/pybind11-prefix/src/pybind11-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "H:/shliao/DCVC/DCVC-DC/src/build/3rdparty/pybind11/pybind11-download/pybind11-prefix/src/pybind11-stamp${cfgdir}") # cfgdir has leading slash
endif()
