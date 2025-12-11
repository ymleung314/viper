get_target_property(DISKANN_S_INCLUDE_DIRS diskann_s INCLUDE_DIRECTORIES)

link_libraries(diskann_s)
include_directories(${DISKANN_S_INCLUDE_DIRS})

add_executable(shard_data shard_data.cpp)
