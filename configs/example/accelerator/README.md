# Custom Accelerator

Custom accelerator model:
 * 16x16 systolic array,
 * DMA engine,
 * RV64GCV vector CPU

## 1. Build

Build the flash-attention operator
requires a riscv64 bare-metal toolchain and a gem5 build
(`build/RISCV/gem5.opt`). `M5_PATH` must point at the gem5 root:

```sh
export PATH=/path/to/riscv64/bin:$PATH
export M5_PATH=/path/to/gem5
cd /path/to/gem5/configs/example/accelerator/flash_attention
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=cmake/riscv64.cmake
cmake --build build --target gen_data flash_accelerator -j8
```

The tile count is the `FLASH_TILES` variable
(`sw/tools/CMakeLists.txt`). After tuning it, regenerate the
workload data:

```sh
cmake -B build -DFLASH_TILES=32
cmake --build build --target gen_data -j8
```

Target `gen_data` regenerates `build/generated/flash_data.bin` and
`build/generated/O_ref.bin`.

## 2. Run the simulation

```sh
cd /path/to/gem5
MAX_TICK=5000000000 build/RISCV/gem5.opt \
  --outdir=/path/to/fa_out \
  configs/example/accelerator/accelerator.py \
  --elf configs/example/accelerator/flash_attention/build/workload/flash_accelerator.elf \
  --data configs/example/accelerator/flash_attention/build/generated/flash_data.bin
```

The run ends by itself (`m5_exit`); `MAX_TICK` is a safety cap only.
For a timeline trace add:

```sh
  --debug-flags=MatMulExec,DmaExec,O3PipeView,UscopeView,Exec \
  --debug-file=/path/to/fa_out/trace.out
```

## 3. Validate against the reference

```sh
cd /path/to/gem5/configs/example/accelerator/flash_attention
python3 sw/tools/validate_flash.py \
  --sim /path/to/fa_out/O_dump.bin \
  --ref build/generated/O_ref.bin \
  --shape 16 128 --tol 1e-6
```