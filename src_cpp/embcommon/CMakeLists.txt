## Copyright 2009-2021 Intel Corporation
## SPDX-License-Identifier: Apache-2.0


set(TASKING_TBB ON)

ADD_SUBDIRECTORY(sys)
ADD_SUBDIRECTORY(math)
ADD_SUBDIRECTORY(simd)
ADD_SUBDIRECTORY(lexers)
ADD_SUBDIRECTORY(tasking)

add_library(embcommon INTERFACE)
target_include_directories(embcommon INTERFACE ${EMBREE_INCLUDE_DIRS})
target_link_libraries(embcommon INTERFACE sys math simd lexers tasking embree)
target_compile_definitions(embcommon INTERFACE TASKING_TBB)