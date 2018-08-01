cmake_minimum_required(VERSION 2.8)
project(p0)

set(CMAKE_C_COMPILER gcc)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O2")
if (DEBUG)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O0 -g")
endif ()

message(STATUS "PROJECT_SOURCE_DIR: " ${PROJECT_SOURCE_DIR})

set(DRIVER ${PROJECT_SOURCE_DIR}/sdriver.pl)
set(MSHREF ${PROJECT_SOURCE_DIR}/mshref)
set(MSHARGS "-p")

include_directories(
        "${PROJECT_SOURCE_DIR}"
)

add_executable(fib fib.c)
add_executable(psh psh.c util.c)
add_executable(handle util.c handle.c)
add_executable(mykill mykill.c util.c)
add_executable(msh msh.c util.c jobs.c log.c)

if (SHELL_TESTS)
    enable_testing()

    function(shell_test trace_file)
        set(test_name "${PROJECT_SOURCE_DIR}/${trace_file}.txt")
        add_test(NAME "${trace_file}"
                COMMAND sh -c "${DRIVER} -t ${test_name} -s $<TARGET_FILE:msh> -a ${MSHARGS}"
                )
        add_test(NAME "r${trace_file}"
                COMMAND sh -c "${DRIVER} -t ${test_name} -s ${MSHREF} -a ${MSHARGS}"
                )
    endfunction(shell_test)

    foreach (i RANGE 1 16)
        if (${i} LESS 10)
            shell_test(trace0${i})
        else ()
            shell_test(trace${i})
        endif ()
    endforeach ()
endif (SHELL_TESTS)