# getting variables
include(cmake/init.cmake)
add_definitions(-DTRAINING_IMAGES_FILE="${RES_DIR}/train-images.idx3-ubyte")
add_definitions(-DTRAINING_LABELS_FILE="${RES_DIR}/train-labels.idx1-ubyte")
add_definitions(-DKNN_EUCLID)
add_definitions(-DKNN_NUM_OF_THREADS=20 )
add_definitions(-DTRAIN_SET_PERCENT=0.75)
add_definitions(-DTEST_SET_PERCENT=0.20)
add_definitions(-DVALIDATION_SET_PERCENT=0.05)
message(STATUS "TRAINING_IMAGES_FILE: ${RES_DIR}/train-images.idx3-ubyte")
message(STATUS "TRAINING_LABELS_FILE: ${RES_DIR}/train-labels.idx1-ubyte")
message(STATUS "KNN_DISTANCE: KNN_EUCLID")
add_subdirectory(${LIB_DIR}/data_model)
add_subdirectory(${LIB_DIR}/mnist)
add_subdirectory(${LIB_DIR}/knn)
add_subdirectory(${LIB_DIR}/k-means)
add_subdirectory(${APPS_DIR})