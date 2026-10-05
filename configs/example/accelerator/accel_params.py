"""Shared parameters of the Vector CPU in accelerator platform"""

# Clock domain
CLK = "1GHz"

# The vector CPU: RV64GCV with RVV 1.0, VLEN = 1024 bit,
# pseudo dual-issue: 1 scalar + 1 vector operation per cycle
VLEN = 1024
ISSUE_WIDTH = 2

# DRAM base address
DRAM_BASE = 0x80000000

# SRAM timing
MEM_LATENCY = "1ns"
MEM_BANDWIDTH = "128GiB/s"

# gem5's m5_exit code handling mask
EXIT_CODE_MASK = 0xFFFFFFFF
