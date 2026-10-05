/*
 * Minimal libc shim for the bare-metal firmware: memset
 */

extern "C" void *memset(void *dest, int ch, unsigned long count) {
  auto *bytes = static_cast<volatile unsigned char *>(dest);
  for (unsigned long i = 0; i < count; ++i) {
    bytes[i] = static_cast<unsigned char>(ch);
  }
  return dest;
}
