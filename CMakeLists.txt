cmake_minimum_required(VERSION 2.4)
PROJECT(SuperResolution)
cmake_policy(SET CMP0003 NEW)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)

FIND_PACKAGE(ITK)
IF(ITK_FOUND)
   INCLUDE(${ITK_USE_FILE})
ELSE(ITK_FOUND)
	MESSAGE(FATAL_ERROR "itk not found")
ENDIF(ITK_FOUND)


#FIND_PACKAGE(VTK)
#IF(VTK_FOUND)
#  INCLUDE(${VTK_USE_FILE})
#ELSE(VTK_FOUND)
#	MESSAGE(FATAL_ERROR "VTK NOT FOUND")
#ENDIF(VTK_FOUND)

  # find ANTS includes
  message("ANTs_SOURCE_DIR=${ANTs_SOURCE_DIR}")
  include_directories(${BOOST_INCLUDE_DIR})
  include_directories(${ANTs_SOURCE_DIR}/Temporary)
  include_directories(${ANTs_SOURCE_DIR}/Tensor)
  include_directories(${ANTs_SOURCE_DIR}/Utilities)
  include_directories(${ANTs_SOURCE_DIR}/Examples)
  include_directories(${ANTs_SOURCE_DIR}/ImageRegistration)
  set(ANTS_LIBS antsUtilities)


INCLUDE_DIRECTORIES( ${PROJECT_SOURCE_DIR}/inc )
ADD_SUBDIRECTORY (${PROJECT_SOURCE_DIR}/src )
ADD_SUBDIRECTORY (${PROJECT_SOURCE_DIR}/Utilities )



#ADD_EXECUTABLE(estimate_tensors EstimateTensors.cpp)
#TARGET_LINK_LIBRARIES(estimate_tensors ${ITK_LIBRARIES} ${VTK_LIBRARIES} SuperResolution)

ADD_EXECUTABLE(stkEstimateTensors STKEstimateTensors.cpp)
TARGET_LINK_LIBRARIES(stkEstimateTensors ${ITK_LIBRARIES} ${VTK_LIBRARIES} SuperResolution)

ADD_EXECUTABLE(stkEstimateTensor_DispField STKEstimateTensors_DispField.cpp)
TARGET_LINK_LIBRARIES(stkEstimateTensor_DispField ${ITK_LIBRARIES} SuperResolution) 

ADD_EXECUTABLE(StupidTensorEstimation StupidTensorEstimation.cpp)
TARGET_LINK_LIBRARIES(StupidTensorEstimation ${ITK_LIBRARIES} SuperResolution) 

ADD_EXECUTABLE(stkEstimateTensors_v2 STKTensorEstimation_v2.cpp)
TARGET_LINK_LIBRARIES(stkEstimateTensors_v2 ${ITK_LIBRARIES} ${VTK_LIBRARIES} SuperResolution)

ADD_EXECUTABLE(STKEstimateTensors_CWLS STKEstimateTensors_CWLS.cpp)
TARGET_LINK_LIBRARIES(STKEstimateTensors_CWLS ${ITK_LIBRARIES} ${VTK_LIBRARIES} SuperResolution)
