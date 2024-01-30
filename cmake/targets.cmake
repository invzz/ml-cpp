# getting variables
include(cmake/init.cmake)

add_subdirectory(${LIB_DIR}/data_model)
add_subdirectory(${LIB_DIR}/mnist)
add_subdirectory(${LIB_DIR}/knn)
add_subdirectory(${LIB_DIR}/thread_queue)
add_subdirectory(${APPS_DIR})