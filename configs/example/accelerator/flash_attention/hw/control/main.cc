/*
 * Control CPU entry point
 */

#include "accelerator_config.hh"

extern "C" void control_main();

int main() {
  control_main();

  for (;;) {
    __asm__ volatile("wfi");
  }
}

extern "C" void control_entry() { main(); }
