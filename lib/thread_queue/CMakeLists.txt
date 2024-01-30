set(LIB_NAME thread_queue)

project(${LIB_NAME})

add_definitions(-DTRAINING_IMAGES_FILE="${RES_DIR}/train-images.idx3-ubyte")
add_definitions(-DTRAINING_LABELS_FILE="${RES_DIR}/train-labels.idx1-ubyte")
add_definitions(-DKNN_EUCLID)
message(STATUS "TRAINING_IMAGES_FILE: ${RES_DIR}/train-images.idx3-ubyte")
message(STATUS "TRAINING_LABELS_FILE: ${RES_DIR}/train-labels.idx1-ubyte")
message(STATUS "KNN_DISTANCE: KNN_EUCLID")
include_directories(
    BEFORE
    ${PROJECT_SOURCE_DIR}/inc
)

#set(DEPENDSON ${DEPENDSON} mnist_handler knn data_model)
set(INC ${PROJECT_SOURCE_DIR}/inc/${LIB_NAME}.hh )
add_library(${LIB_NAME} INTERFACE ${INC} )

if(DEFINED DEPENDSON)
    MESSAGE(STATUS "${LIB_NAME} resolving dependencies...: ${DEPENDSON}")
    target_link_libraries(${LIB_NAME} ${DEPENDSON}) 
endif()