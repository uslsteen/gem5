/*
 * Linker script for the two-CPU whole-accelerator firmware.
 */

OUTPUT_ARCH("riscv")
ENTRY(_start)

_stack_size_control = 0x10000;   /* 64 KiB, Control CPU */
_stack_size_vector = 0x4000;     /* 16 KiB, Vector CPU (in the SRAM) */
/* The Control CPU temporary stack in the DRAM reset stub (start.S) */
_boot_stack_control = 0x110E0000;

SECTIONS
{
    . = 0x80000000;

    /* DRAM reset stub; the far-jump dispatch slots live inside
     * .text._start */
    .text.boot : {
        *(.text._start)
    }

    /* --- the Vector CPU SRAM code (fetched from the SRAM directly
     * --- via the iobus; the sync flags at the SRAM base stay clear
     * --- of the code) --- */
    . = 0x11001000;
    .sram.text : {
        *(.sram_text .sram_text.*)
    }

    /* --- the Vector CPU zero-initialised state. NOT cleared at
     * --- boot: the SRAM boots zeroed in gem5, and the zero-init
     * --- avoids memset calls from the SRAM code. --- */
    . = 0x11002000;
    .sram.bss (NOLOAD) : {
        . = ALIGN(8);
        __sram_bss_start = .;
        *(.sram_bss .sram_bss.*)
        . = ALIGN(8);
        __sram_bss_end = .;
    }

    /* --- the Control CPU: all of its code, constants, data, bss and
     * --- stack in the SRAM workspace. Everything here was previously in
     * --- DRAM; the far jump from the stub into control_boot needs the
     * --- 64-bit slot (medany cannot reach 0x11010000 from 0x80000000). */
    . = 0x11010000;
    .control.text : {
        *(.text .text.*)
    }

    .control.rodata : {
        *(.rodata .rodata.*)
        *(.srodata .srodata.*)
    }

    .control.data : {
        *(.data .data.*)
        *(.sdata .sdata.*)
    }

    .control.bss (NOLOAD) : {
        . = ALIGN(8);
        __bss_start_control = .;
        *(.sbss .sbss.*)
        *(.bss .bss.*)
        *(COMMON)
        . = ALIGN(8);
        __bss_end_control = .;
    }

    . = ALIGN(16);
    . += _stack_size_control;
    _stack_top_control = .;

    /* --- the Vector CPU SRAM stack (16 KiB at the workspace top) --- */
    _stack_top_vector = 0x11100000;

    _end = .;
}
