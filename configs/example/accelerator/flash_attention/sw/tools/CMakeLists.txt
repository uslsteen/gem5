set(FLASH_TILES 2 CACHE STRING "Number of flash tiles")
set(FLASH_REGIME prefill CACHE STRING "prefill|decode")
set(FLASH_CAUSAL OFF CACHE BOOL "Apply causal boundary mask to the last tile")

set(GEN_STAMP ${CMAKE_BINARY_DIR}/flash_gen_params.txt)
file(GENERATE OUTPUT ${GEN_STAMP}
     CONTENT "${FLASH_TILES} ${FLASH_REGIME} ${FLASH_CAUSAL}")

set(FLASH_CAUSAL_ARG "")
if(FLASH_CAUSAL)
    set(FLASH_CAUSAL_ARG --causal)
endif()

add_custom_command(
    OUTPUT ${GEN_DIR}/flash_data.hh
           ${GEN_DIR}/flash_data.bin
           ${GEN_DIR}/O_ref.bin
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/gen_flash.py
            --tiles ${FLASH_TILES} --regime ${FLASH_REGIME}
            ${FLASH_CAUSAL_ARG}
            --gen-input --gen-reference --emit-bin
            --out ${GEN_DIR}/flash_data.hh
            --input-bin ${GEN_DIR}/flash_data.bin
            --reference-bin ${GEN_DIR}/O_ref.bin
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/gen_flash.py
            ${CMAKE_CURRENT_SOURCE_DIR}/consts.py
            ${CMAKE_CURRENT_SOURCE_DIR}/common.py
            ${GEN_STAMP}
    COMMENT "Generating flash data (tiles=${FLASH_TILES}, ${FLASH_REGIME}, causal=${FLASH_CAUSAL})"
    VERBATIM)

add_custom_command(
    OUTPUT ${GEN_DIR}/exp_coeffs.h
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/gen_exp_coeffs.py
            --degree 5
            --out ${GEN_DIR}/exp_coeffs.h
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/gen_exp_coeffs.py
    COMMENT "Generating the 2^r minimax coefficients (exp_coeffs.h)"
    VERBATIM)

add_custom_target(gen_data
    DEPENDS ${GEN_DIR}/flash_data.hh ${GEN_DIR}/flash_data.bin
            ${GEN_DIR}/O_ref.bin
            ${GEN_DIR}/exp_coeffs.h)
