set(APP_NAME etl)

project(${APP_NAME})

add_definitions(-DTRAINING_IMAGES_FILE="${RES_DIR}/train-images.idx3-ubyte")
add_definitions(-DTRAINING_LABELS_FILE="${RES_DIR}/train-labels.idx1-ubyte")
add_definitions(-DKNN_EUCLID)
add_definitions(-DNUMBER_OF_THREADS=50 )
message(STATUS "TRAINING_IMAGES_FILE: ${RES_DIR}/train-images.idx3-ubyte")
message(STATUS "TRAINING_LABELS_FILE: ${RES_DIR}/train-labels.idx1-ubyte")
message(STATUS "KNN_DISTANCE: KNN_EUCLID")
include_directories(
    BEFORE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/inc
    ${LIB_DIR}/mnist/inc
    ${LIB_DIR}/knn/inc
    ${LIB_DIR}/data_model/inc
)

set(DEPENDSON ${DEPENDSON} mnist_handler knn data_model)
set(SRC ${PROJECT_SOURCE_DIR}/src/main.cc )
set(INC ${PROJECT_SOURCE_DIR}/inc/main.hh )
add_executable(${APP_NAME} ${SRC} ${INC} )
if(DEFINED DEPENDSON)
    MESSAGE(STATUS "${APP_NAME} resolbing dependencies...: ${DEPENDSON}")
    target_link_libraries(${APP_NAME} ${DEPENDSON}) 
endif()