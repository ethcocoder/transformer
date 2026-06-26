cmake_minimum_required(VERSION 3.20)
project(aurelis VERSION 0.1.0 LANGUAGES C CXX)

include(GNUInstallDirs)
include(CPack)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(AURELIS_USE_BLAS "Link OpenBLAS/MKL for sgemm" ON)
option(AURELIS_BUILD_PYTHON "Build pybind11 module" OFF)
option(AURELIS_BUILD_TESTS "Build unit tests" ON)

# --- C kernel library ---
set(AURELIS_C_SOURCES
    src/c/scan_blelloch.c
    src/c/recurrence_diag.c
    src/c/matmul_blas.c
    src/c/linalg_small.c
    src/c/orthogonal.c
    src/c/tensor_ring.c
    src/c/quant_int4_int8.c
)

add_library(aurelis_c STATIC ${AURELIS_C_SOURCES})
target_include_directories(aurelis_c PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src/c
)

if(AURELIS_USE_BLAS)
    find_package(BLAS)
    if(BLAS_FOUND)
        target_link_libraries(aurelis_c PUBLIC BLAS::BLAS)
        target_compile_definitions(aurelis_c PUBLIC AURELIS_USE_BLAS=1)
        message(STATUS "Aurelis: using BLAS for matmul")
    else()
        message(STATUS "Aurelis: BLAS not found — using naive matmul fallback")
    endif()
endif()

if(MSVC)
    target_compile_options(aurelis_c PRIVATE /W4)
else()
    target_compile_options(aurelis_c PRIVATE -Wall -Wextra -O2)
endif()

# --- C++ core library ---
set(AURELIS_CORE_SOURCES
    src/core/tensor.cpp
    src/core/autodiff.cpp
    src/core/spectral.cpp
    src/core/state_partition.cpp
    src/core/logging.cpp
    src/core/profiler.cpp
)

add_library(aurelis_core STATIC ${AURELIS_CORE_SOURCES})
target_include_directories(aurelis_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(aurelis_core PUBLIC aurelis_c)

# Link filesystem library for older GCC versions
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT MINGW)
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
        target_link_libraries(aurelis_core PUBLIC stdc++fs)
    endif()
endif()

if(MSVC)
    target_compile_options(aurelis_core PRIVATE /W4)
else()
    target_compile_options(aurelis_core PRIVATE -Wall -Wextra -O2)
endif()

# --- LENS (Phase I) ---
set(AURELIS_LENS_SOURCES
    src/lens/activations.cpp
    src/lens/linear.cpp
    src/lens/mssp.cpp
    src/lens/ietcf.cpp
    src/lens/fwse.cpp
    src/lens/csc.cpp
    src/lens/mgp.cpp
    src/lens/spi.cpp
    src/lens/osh.cpp
    src/lens/lens_layer.cpp
    src/lens/loss.cpp
    src/lens/lens_model.cpp
    src/lens/bpe.cpp
)

add_library(aurelis_lens STATIC ${AURELIS_LENS_SOURCES})
target_include_directories(aurelis_lens PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(aurelis_lens PUBLIC aurelis_core)

if(MSVC)
    target_compile_options(aurelis_lens PRIVATE /W4)
else()
    target_compile_options(aurelis_lens PRIVATE -Wall -Wextra -O2)
endif()

# --- AUREUM (Phase II) ---
set(AURELIS_AUREUM_SOURCES
    src/aureum/ese.cpp
    src/aureum/cq.cpp
    src/aureum/kmg.cpp
    src/aureum/dtc.cpp
    src/aureum/ccc.cpp
    src/aureum/pii.cpp
)

add_library(aurelis_aureum STATIC ${AURELIS_AUREUM_SOURCES})
target_include_directories(aurelis_aureum PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(aurelis_aureum PUBLIC aurelis_lens)

if(MSVC)
    target_compile_options(aurelis_aureum PRIVATE /W4)
else()
    target_compile_options(aurelis_aureum PRIVATE -Wall -Wextra -O2)
endif()

# --- ARC (Phase III) ---
set(AURELIS_ARC_SOURCES
    src/arc/ale.cpp
    src/arc/rcvg.cpp
    src/arc/mc_fws.cpp
    src/arc/rse.cpp
    src/arc/ccrc.cpp
    src/arc/roi.cpp
    src/arc/ask_r.cpp
)

add_library(aurelis_arc STATIC ${AURELIS_ARC_SOURCES})
target_include_directories(aurelis_arc PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(aurelis_arc PUBLIC aurelis_aureum)

if(MSVC)
    target_compile_options(aurelis_arc PRIVATE /W4)
else()
    target_compile_options(aurelis_arc PRIVATE -Wall -Wextra -O2)
endif()

# --- TESSERAE (Phase IV) ---
set(AURELIS_TESSERAE_SOURCES
    src/tesserae/inws.cpp
    src/tesserae/trfc.cpp
    src/tesserae/epqe.cpp
    src/tesserae/eds.cpp
    src/tesserae/efd.cpp
    src/tesserae/cvk.cpp
    src/tesserae/p4il.cpp
)

add_library(aurelis_tesserae STATIC ${AURELIS_TESSERAE_SOURCES})
target_include_directories(aurelis_tesserae PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(aurelis_tesserae PUBLIC aurelis_arc)

if(MSVC)
    target_compile_options(aurelis_tesserae PRIVATE /W4)
else()
    target_compile_options(aurelis_tesserae PRIVATE -Wall -Wextra -O2)
endif()

# --- CONVERGENCE (Phase V) ---
set(AURELIS_CONVERGENCE_SOURCES
    src/convergence/agi_bus.cpp
    src/convergence/efe.cpp
)

add_library(aurelis_convergence STATIC ${AURELIS_CONVERGENCE_SOURCES})
target_include_directories(aurelis_convergence PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(aurelis_convergence PUBLIC aurelis_tesserae)

if(MSVC)
    target_compile_options(aurelis_convergence PRIVATE /W4)
else()
    target_compile_options(aurelis_convergence PRIVATE -Wall -Wextra -O2)
endif()

add_executable(train_lens src/train/train_lens.cpp)
target_link_libraries(train_lens PRIVATE aurelis_lens)

# --- C ABI shared library for Python / FFI ---
add_library(aurelis_capi SHARED src/c/trainer_abi.cpp)
target_include_directories(aurelis_capi PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(aurelis_capi PRIVATE aurelis_lens)
if(MSVC)
    target_compile_options(aurelis_capi PRIVATE /W4)
else()
    target_compile_options(aurelis_capi PRIVATE -Wall -Wextra -O2 -fvisibility=hidden)
    set_target_properties(aurelis_capi PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
    )
endif()

# --- Install rules for packaging / release ---
install(TARGETS
    aurelis_c
    aurelis_core
    aurelis_lens
    aurelis_aureum
    aurelis_arc
    aurelis_tesserae
    aurelis_convergence
    train_lens
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES README.md DESTINATION ${CMAKE_INSTALL_DOCDIR})

set(CPACK_PACKAGE_NAME "aurelis")
set(CPACK_PACKAGE_VENDOR "Aurelis")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Aurelis release package")
set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
set(CPACK_GENERATOR "ZIP;TGZ")
set(CPACK_SOURCE_GENERATOR "ZIP;TGZ")
set(CPACK_PACKAGE_FILE_NAME "aurelis-${PROJECT_VERSION}-release")
set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")

# --- Tests ---
if(AURELIS_BUILD_TESTS)
    enable_testing()

    add_executable(test_scan_c tests/c/test_scan.c)
    target_link_libraries(test_scan_c PRIVATE aurelis_c)
    add_test(NAME scan_c COMMAND test_scan_c)

    add_executable(test_tensor_cpp tests/cpp/test_tensor.cpp)
    target_link_libraries(test_tensor_cpp PRIVATE aurelis_core)
    add_test(NAME tensor_cpp COMMAND test_tensor_cpp)

    add_executable(test_phase0_cpp tests/cpp/test_phase0.cpp)
    target_link_libraries(test_phase0_cpp PRIVATE aurelis_core)
    add_test(NAME phase0_cpp COMMAND test_phase0_cpp)

    add_executable(test_fwse_spectral tests/cpp/test_fwse_spectral.cpp)
    target_link_libraries(test_fwse_spectral PRIVATE aurelis_lens)
    add_test(NAME fwse_spectral COMMAND test_fwse_spectral)

    add_executable(test_mgp tests/cpp/test_mgp.cpp)
    target_link_libraries(test_mgp PRIVATE aurelis_lens)
    add_test(NAME mgp COMMAND test_mgp)

    add_executable(test_lens_forward tests/cpp/test_lens_forward.cpp)
    target_link_libraries(test_lens_forward PRIVATE aurelis_lens)
    add_test(NAME lens_forward COMMAND test_lens_forward)

    add_executable(test_ietcf_only tests/cpp/test_ietcf_only.cpp)
    target_link_libraries(test_ietcf_only PRIVATE aurelis_lens)
    add_test(NAME ietcf_only COMMAND test_ietcf_only)

    add_executable(test_nan_debug tests/cpp/test_nan_debug.cpp)
    target_link_libraries(test_nan_debug PRIVATE aurelis_lens)
    add_test(NAME nan_debug COMMAND test_nan_debug)

    add_executable(test_production_suite tests/cpp/test_production_suite.cpp)
    target_link_libraries(test_production_suite PRIVATE aurelis_lens)
    add_test(NAME production_suite COMMAND test_production_suite)

    add_executable(test_aureum tests/cpp/test_aureum.cpp)
    target_link_libraries(test_aureum PRIVATE aurelis_aureum)
    add_test(NAME aureum COMMAND test_aureum)

    add_executable(test_arc tests/cpp/test_arc.cpp)
    target_link_libraries(test_arc PRIVATE aurelis_arc)
    add_test(NAME arc COMMAND test_arc)

    add_executable(test_convergence tests/cpp/test_convergence.cpp)
    target_link_libraries(test_convergence PRIVATE aurelis_convergence)
    add_test(NAME convergence COMMAND test_convergence)
endif()

# --- Python bindings (optional) ---
if(AURELIS_BUILD_PYTHON)
    find_package(Python COMPONENTS Interpreter Development REQUIRED)
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(aurelis_py python/aurelis/_core.cpp)
    target_link_libraries(aurelis_py PRIVATE aurelis_core)
    target_include_directories(aurelis_py PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
endif()
