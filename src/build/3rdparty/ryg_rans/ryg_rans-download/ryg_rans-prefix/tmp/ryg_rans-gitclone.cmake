# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

if(EXISTS "H:/shliao/DCVC/DCVC-DC/src/build/3rdparty/ryg_rans/ryg_rans-download/ryg_rans-prefix/src/ryg_rans-stamp/ryg_rans-gitclone-lastrun.txt" AND EXISTS "H:/shliao/DCVC/DCVC-DC/src/build/3rdparty/ryg_rans/ryg_rans-download/ryg_rans-prefix/src/ryg_rans-stamp/ryg_rans-gitinfo.txt" AND
  "H:/shliao/DCVC/DCVC-DC/src/build/3rdparty/ryg_rans/ryg_rans-download/ryg_rans-prefix/src/ryg_rans-stamp/ryg_rans-gitclone-lastrun.txt" IS_NEWER_THAN "H:/shliao/DCVC/DCVC-DC/src/build/3rdparty/ryg_rans/ryg_rans-download/ryg_rans-prefix/src/ryg_rans-stamp/ryg_rans-gitinfo.txt")
  message(STATUS
    "Avoiding repeated git clone, stamp file is up to date: "
    "'H:/shliao/DCVC/DCVC-DC/src/build/3rdparty/ryg_rans/ryg_rans-download/ryg_rans-prefix/src/ryg_rans-stamp/ryg_rans-gitclone-lastrun.txt'"
  )
  return()
endif()

execute_process(
  COMMAND ${CMAKE_COMMAND} -E rm -rf "H:/shliao/DCVC/DCVC-DC/src/build/3rdparty/ryg_rans/ryg_rans-src"
  RESULT_VARIABLE error_code
)
if(error_code)
  message(FATAL_ERROR "Failed to remove directory: 'H:/shliao/DCVC/DCVC-DC/src/build/3rdparty/ryg_rans/ryg_rans-src'")
endif()

# try the clone 3 times in case there is an odd git clone issue
set(error_code 1)
set(number_of_tries 0)
while(error_code AND number_of_tries LESS 3)
  execute_process(
    COMMAND "F:/git/Git/cmd/git.exe"
            clone --no-checkout --depth 1 --no-single-branch --config "advice.detachedHead=false" "https://github.com/rygorous/ryg_rans.git" "ryg_rans-src"
    WORKING_DIRECTORY "H:/shliao/DCVC/DCVC-DC/src/build/3rdparty/ryg_rans"
    RESULT_VARIABLE error_code
  )
  math(EXPR number_of_tries "${number_of_tries} + 1")
endwhile()
if(number_of_tries GREATER 1)
  message(STATUS "Had to git clone more than once: ${number_of_tries} times.")
endif()
if(error_code)
  message(FATAL_ERROR "Failed to clone repository: 'https://github.com/rygorous/ryg_rans.git'")
endif()

execute_process(
  COMMAND "F:/git/Git/cmd/git.exe"
          checkout "c9d162d996fd600315af9ae8eb89d832576cb32d" --
  WORKING_DIRECTORY "H:/shliao/DCVC/DCVC-DC/src/build/3rdparty/ryg_rans/ryg_rans-src"
  RESULT_VARIABLE error_code
)
if(error_code)
  message(FATAL_ERROR "Failed to checkout tag: 'c9d162d996fd600315af9ae8eb89d832576cb32d'")
endif()

set(init_submodules TRUE)
if(init_submodules)
  execute_process(
    COMMAND "F:/git/Git/cmd/git.exe" 
            submodule update --recursive --init 
    WORKING_DIRECTORY "H:/shliao/DCVC/DCVC-DC/src/build/3rdparty/ryg_rans/ryg_rans-src"
    RESULT_VARIABLE error_code
  )
endif()
if(error_code)
  message(FATAL_ERROR "Failed to update submodules in: 'H:/shliao/DCVC/DCVC-DC/src/build/3rdparty/ryg_rans/ryg_rans-src'")
endif()

# Complete success, update the script-last-run stamp file:
#
execute_process(
  COMMAND ${CMAKE_COMMAND} -E copy "H:/shliao/DCVC/DCVC-DC/src/build/3rdparty/ryg_rans/ryg_rans-download/ryg_rans-prefix/src/ryg_rans-stamp/ryg_rans-gitinfo.txt" "H:/shliao/DCVC/DCVC-DC/src/build/3rdparty/ryg_rans/ryg_rans-download/ryg_rans-prefix/src/ryg_rans-stamp/ryg_rans-gitclone-lastrun.txt"
  RESULT_VARIABLE error_code
)
if(error_code)
  message(FATAL_ERROR "Failed to copy script-last-run stamp file: 'H:/shliao/DCVC/DCVC-DC/src/build/3rdparty/ryg_rans/ryg_rans-download/ryg_rans-prefix/src/ryg_rans-stamp/ryg_rans-gitclone-lastrun.txt'")
endif()
