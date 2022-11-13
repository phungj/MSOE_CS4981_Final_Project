cmake_minimum_required(VERSION 3.10.2)	#Specify the minimum CMake version
project(small_mc)	#Specify the name of the project
find_package(CUDA REQUIRED)	#find the CUDA libraries
include_directories(${CUDA_INCLUDE_DIRS})	#Specify the CUDA include directories for the project
cuda_add_executable("${PROJECT_NAME}" main.cu)	#create an executable from the given CUDA files

#specify any additional libraries here 
target_link_libraries("${PROJECT_NAME}")
set(CMAKE_CXX_FLAGS_DEBUG  "-g")

