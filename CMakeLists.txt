PROJECT(BrainAlignment)

FIND_PACKAGE(ITK REQUIRED)

IF(ITK_FOUND)
	INCLUDE(${ITK_USE_FILE})
ELSE(ITK_FOUND)
	MESSAGE(FATAL_ERROR"ITK not found. Please set ITK_DIR.")
ENDIF(ITK_FOUND)

SET (BrainAlignment_INCLUDE_DIR ${BrainAlignment_SOURCE_DIR}/include)
SET (BrainAlignment_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

SET(BrainAlignment_HEADERS 
   ${BrainAlignment_INCLUDE_DIR}/BrainAlignmentFilter.h
)

SET(BrainAlignment_SRCS 
${BrainAlignment_SOURCE_DIR}/main.cxx
${BrainAlignment_SOURCE_DIR}/BrainAlignmentFilter.cxx
)

INCLUDE_DIRECTORIES("${BrainAlignment_INCLUDE_DIR}")

ADD_EXECUTABLE(BrainAlignment ${BrainAlignment_HEADERS} ${BrainAlignment_SRCS})

TARGET_LINK_LIBRARIES (BrainAlignment   ${ITK_LIBRARIES})	

