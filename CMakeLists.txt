cmake_minimum_required(VERSION 3.1...3.28)

project(MNIST VERSION  0.5 LANGUAGES CXX)



include(cmake/automate-vcpkg.cmake)
vcpkg_bootstrap()
vcpkg_integrate_install()

IF (WIN32)
  # set stuff for windows
  find_package(pthread REQUIRED)
ELSE()
  # set stuff for other systems
ENDIF()
    
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/dist)

include(cmake/init.cmake)
include(cmake/folders.cmake)
include(cmake/targets.cmake)

message(STATUS "RES_DIR: " ${PROJECT_SOURCE_DIR}/resources/)
