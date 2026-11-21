/* Freestanding string.h for the demo's wasm32 build (demo/wasm/shim.c). */
#ifndef DEMO_STRING_H
#define DEMO_STRING_H

#include <stddef.h>

void *memcpy(void *restrict d, const void *restrict s, size_t n);
void *memset(void *d, int c, size_t n);
int memcmp(const void *a, const void *b, size_t n);
size_t strlen(const char *s);

#endif
