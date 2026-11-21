/* The four string.h functions src/menu.c uses (and the compiler may call for struct copies),
 * for the freestanding wasm32 build: there is no libc in the module. */
#include <stddef.h>

#include "string.h"

void *memcpy(void *restrict d, const void *restrict s, size_t n) {
    unsigned char *a = d;
    const unsigned char *b = s;
    while (n--) {
        *a++ = *b++;
    }
    return d;
}

void *memset(void *d, int c, size_t n) {
    unsigned char *a = d;
    while (n--) {
        *a++ = (unsigned char)c;
    }
    return d;
}

int memcmp(const void *x, const void *y, size_t n) {
    const unsigned char *a = x, *b = y;
    for (; n; n--, a++, b++) {
        if (*a != *b) {
            return *a < *b ? -1 : 1;
        }
    }
    return 0;
}

size_t strlen(const char *s) {
    const char *p = s;
    while (*p) {
        p++;
    }
    return (size_t)(p - s);
}
