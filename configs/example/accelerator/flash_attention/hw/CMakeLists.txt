set(HW_DEPS
    ${BENCH_SRC}/sw/include/config.h
    ${BENCH_SRC}/sw/include/accelerator_config.hh
    ${GEN_DIR}/exp_coeffs.h)

# Accelerator firmware: hart 0 = Control CPU, hart 1 = Vector CPU softmax. 
# Booted by configs/example/accelerator/accelerator.py (the workload-agnostic platform).
add_executable(flash_accelerator
    control/main.cc
    control/control_driver.cc
    vector/vector_kernel.cc
    vector/rvv_softmax.cc
    ${BENCH_SRC}/sw/runtime/runtime.cc
    control/start.S
    ${M5_PATH}/util/m5/src/abi/riscv/m5op.S)

set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/control/start.S
    ${M5_PATH}/util/m5/src/abi/riscv/m5op.S
    PROPERTIES LANGUAGE C COMPILE_FLAGS "-x assembler-with-cpp")

target_compile_options(flash_accelerator PRIVATE
    -g -O3 -march=rv64gcv_zvl1024b -mabi=lp64d -mcmodel=medany)
target_compile_options(flash_accelerator PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions>)


set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/vector/vector_kernel.cc
    PROPERTIES COMPILE_OPTIONS "-fno-tree-vectorize")

target_include_directories(flash_accelerator PRIVATE
    ${BENCH_SRC}/sw/include
    ${GEN_DIR}
    ${M5_PATH}/include)
target_link_options(flash_accelerator PRIVATE
    -nostartfiles
    -Xlinker -T${CMAKE_CURRENT_SOURCE_DIR}/control/link.ld)

set_target_properties(flash_accelerator PROPERTIES
    PREFIX "" SUFFIX ".elf" OUTPUT_NAME "flash_accelerator"
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/workload)

add_dependencies(flash_accelerator gen_data)
