/*
 * gem5 bare-metal RISC-V resets BOTH harts to the single ELF entry
 *
 *   hart 0 = the Control CPU  -> control_boot() -> control_entry() -> main() -> control_main()
 *   hart 1 = the Vector CPU   -> vector_main() (the softmax microkernel
 *
 * There is no HTIF tohost exit here — the m5ops replace it.
 */

    .section .text._start, "ax"
    .globl _start
_start:
    csrr    t0, mhartid
    bnez    t0, .Lvector_hart

    // Control CPU. A temporary SRAM stack
    lui     sp, %hi(_boot_stack_control)
    addi    sp, sp, %lo(_boot_stack_control)

    li      t0, (3 << 13) | (3 << 9)   /* mstatus.FS | mstatus.VS = Dirty */
    csrs    mstatus, t0

    // Absolute jump into control_boot (the SRAM control code)
    .balign 8
    auipc   t0, 0
    ld      t0, 16(t0)
    jr      t0
    .balign 8
    .dword  control_boot

.Lvector_hart:
    li      sp, 0x11100000

    li      t0, (3 << 13) | (3 << 9)
    csrs    mstatus, t0

    // Absolute jump into vector_main
    .balign 8
    auipc   t0, 0
    ld      t0, 16(t0)
    jr      t0
    .balign 8
    .dword  vector_main

.Lpark:
    wfi
    j       .Lpark

    // the Control hart's SRAM-resident boot: stack, .bss clear, hand over to main()
    .section .text.control_boot, "ax"
    .globl control_boot
control_boot:
    la      sp, _stack_top_control

    la      t0, __bss_start_control
    la      t1, __bss_end_control
1:
    bgeu    t0, t1, 3f
    sd      zero, 0(t0)
    addi    t0, t0, 8
    j       1b
3:
    call    control_entry

.Lpark_control:
    wfi
    j       .Lpark_control
