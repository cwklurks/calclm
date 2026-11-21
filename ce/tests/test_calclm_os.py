"""Exercise the real OS model load path with fileioc/RAM stand-ins on the host.

The same model.c is linked into CALCLM. The emulator separately exercises its shared
post-load decoder policy and the exact core.o. No TI-OS or GUI launch is claimed here.
"""
from __future__ import annotations

import ctypes
import subprocess
from pathlib import Path

import pytest

from ce.calclm.tools.make_ld2 import checksum, payload
from ce.calclm.tools.make_vocab import build
from ce.tools.clm import CALCLM, EXPORT_DIR, HOST_SRC, TRACKER_DIR, V2_EXPORT, Export

pytestmark = pytest.mark.skipif(not (V2_EXPORT / "model.bin").exists(), reason="needs v2 export")
OFFSETS = Path(__file__).parent / "fixtures/ld2_offsets_full.json"


@pytest.fixture(scope="module")
def os_model(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("model-os")
    (tmp / "ti").mkdir()
    (tmp / "fileioc.h").write_text('''
#include <stdint.h>
#include <stddef.h>
uint8_t ti_Open(const char *, const char *);
int ti_IsArchived(uint8_t);
void *ti_GetDataPtr(uint8_t);
uint32_t ti_GetSize(uint8_t);
void ti_Close(uint8_t);
int ti_Delete(const char *);
int ti_Resize(size_t, uint8_t);
''')
    (tmp / "ti/vars.h").write_text('#include <stddef.h>\nsize_t os_MemChk(void *);\n')
    (tmp / "memory.h").write_text('''
#include <stdint.h>
#define CALCLM_MEMMAP_H
extern uint8_t test_tbl[], test_act[];
#define MM_TBL test_tbl
#define MM_ACT test_act
#define MM_ACT_LIMIT 20000
''')
    (tmp / "os.c").write_text('''
#include <stdlib.h>
#include <string.h>
#include "model.h"
uint8_t test_tbl[65536] __attribute__((aligned(65536))), test_act[20000];
static struct { char name[9]; const void *data; uint32_t size; int archived; } av[32];
static void *work;
static size_t avail = 60000;
void test_clear(size_t free_ram) { memset(av, 0, sizeof av); free(work); work = 0; avail = free_ram; }
void test_register(const char *name, const void *p, uint32_t size, int archived) {
    for (int i = 1; i < 32; i++) if (!av[i].name[0]) {
        strcpy(av[i].name, name); av[i].data = p; av[i].size = size; av[i].archived = archived; return;
    }
    abort();
}
uint8_t ti_Open(const char *name, const char *mode) {
    for (int i = 1; i < 32; i++) if (!strcmp(av[i].name, name)) return i;
    if (*mode == 'w') { test_register(name, 0, 0, 0); return ti_Open(name, "r"); }
    return 0;
}
int ti_IsArchived(uint8_t h) { return av[h].archived; }
void *ti_GetDataPtr(uint8_t h) { return (void *)av[h].data; }
uint32_t ti_GetSize(uint8_t h) { return av[h].size; }
void ti_Close(uint8_t h) { (void)h; }
int ti_Delete(const char *name) {
    uint8_t h = ti_Open(name, "r");
    if (h) { av[h].name[0] = 0; if (!strcmp(name, "CLMTMP")) { free(work); work = 0; } }
    return 0;
}
int ti_Resize(size_t n, uint8_t h) {
    work = realloc(work, n); av[h].data = work; av[h].size = n; return n;
}
size_t os_MemChk(void *p) { (void)p; return avail; }
int test_enabled(void) { return M.ld2; }
int test_horizons(void) { return M.horizons; }
int test_tmp_size(void) { return ti_GetSize(ti_Open("CLMTMP", "r")); }
int test_workspace_fits(void) {
    return M.mem.vc == work && M.mem.ctx_cap == model_ctx()
        && M.slide == (model_ctx() == CLM_CTX) && M.ld
        && (uint8_t *)M.ld == (uint8_t *)work + CLM_LAYERS * model_ctx() * CLM_HEAD_DIM
        && (uint8_t *)(M.ld + 1) <= (uint8_t *)work + test_tmp_size();
}
int test_generate(unsigned seed, unsigned cap, uint8_t masks, uint8_t greedy, uint16_t *out) {
    int rc = clm_reset(&M, seed, 205, 20, greedy, cap, masks), n = 0;
    if (rc) return rc;
    while ((rc = clm_step(&M)) >= 0) out[n++] = rc;
    return n;
}
''')
    libpath = tmp / "os.so"
    subprocess.run(["gcc", "-std=c11", "-O1", "-Wall", "-Wextra", "-Werror", "-fPIC",
                    "-shared", "-fsanitize=undefined", "-fno-sanitize-recover=all",
                    f"-I{tmp}", f"-I{CALCLM / 'src'}", f"-I{CALCLM / 'core'}",
                    f"-I{TRACKER_DIR}", "-include", str(tmp / "memory.h"),
                    *map(str, HOST_SRC), str(CALCLM / "src/model.c"), str(tmp / "os.c"),
                    "-o", str(libpath)], check=True, capture_output=True)
    lib = ctypes.CDLL(str(libpath))
    lib.test_register.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int]
    lib.test_clear.argtypes = [ctypes.c_size_t]
    lib.model_open.argtypes = [ctypes.c_char_p]
    lib.model_prepare.argtypes = [ctypes.POINTER(ctypes.c_uint8)]
    lib.test_generate.argtypes = [ctypes.c_uint, ctypes.c_uint, ctypes.c_uint8, ctypes.c_uint8,
                                  ctypes.POINTER(ctypes.c_uint16)]
    return lib


def register(lib, data, free_ram=60000, archived=True, export=None):
    ex = export or Export.load(V2_EXPORT)
    lib.test_clear(free_ram)
    keep = []
    items = [(f"CLM{i:02d}", d, True) for i, d in enumerate(ex.appvars)]
    items.append(("CLMVOC", build(ex.path)[0], True))
    if data is not None:
        items.append(("CLMLD2", data, archived))
    for name, raw, arc in items:
        buf = ctypes.create_string_buffer(raw)
        keep.append(buf)
        lib.test_register(name.encode(), buf, len(raw), arc)
    msg = ctypes.create_string_buffer(200)
    assert lib.model_open(msg) == 0, msg.value
    return keep


@pytest.mark.parametrize("free_ram", [60000, 25000])
def test_real_load_allocates_workspace_and_reenables_on_reload(os_model, free_ram):
    lib = os_model
    data = payload(V2_EXPORT, OFFSETS)
    keep = register(lib, data, free_ram)
    ctx = lib.model_ctx()
    assert lib.test_enabled() == lib.test_horizons() == lib.test_workspace_fits() == 1
    assert lib.test_tmp_size() == 6 * ctx * 32 + 2652
    # gfx_Begin clears activation scratch. E and ledgers live in CLMTMP, not VRAM.
    ctypes.memset(ctypes.addressof((ctypes.c_uint8 * 20000).in_dll(lib, "test_act")), 0, 20000)
    fast = ctypes.c_uint8()
    assert lib.model_prepare(ctypes.byref(fast)) == 0
    assert fast.value == 0 and lib.test_enabled() == lib.test_workspace_fits() == 1
    out = (ctypes.c_uint16 * 1024)()
    n = lib.test_generate(2, 1024, 1, 0, out)
    from model.linker import ARMS, generate
    import json
    ex = Export.load(V2_EXPORT)
    want = generate(ex.model_hz, ARMS["LD2"], 2, E=json.loads(OFFSETS.read_text())["E"],
                    cap=1024)["ids"][1:]
    assert list(out[:n]) == want
    assert keep
    lib.model_close()
    assert register(lib, data, free_ram)
    assert lib.test_enabled() == lib.test_workspace_fits() == 1


@pytest.mark.skipif(not (EXPORT_DIR / "model.bin").exists(), reason="needs v1 export")
def test_format_one_ignores_ld2_and_keeps_its_memory_budget(os_model):
    lib = os_model
    keep = register(lib, payload(V2_EXPORT, OFFSETS), export=Export.load(EXPORT_DIR))
    assert lib.test_enabled() == lib.test_horizons() == 0
    assert lib.test_tmp_size() == 6 * lib.model_ctx() * 32
    assert keep


@pytest.mark.parametrize("free_ram", [60000, 25000])
def test_absent_rejected_or_unarchived_ld2_preserves_v21(os_model, free_ram):
    from model.intmask import IntMask
    from model.intref import SamplerConfig, generate

    lib = os_model
    data = payload(V2_EXPORT, OFFSETS)
    wrong = bytearray(data)
    wrong[9] ^= 1
    wrong[-4:] = checksum(wrong[:-4]).to_bytes(4, "little")
    ex = Export.load(V2_EXPORT)
    want = generate(ex.model_hz, [], 31, SamplerConfig(205, 20, False, 1), cap=32,
                    mask=IntMask(ex.meta["vocab"]))[1:]
    context = None
    for optional, archived in [(None, True), (data[:-1], True), (bytes(wrong), True),
                               (data, False)]:
        keep = register(lib, optional, free_ram, archived)
        assert lib.test_enabled() == 0 and lib.test_horizons() == 1
        ctx = lib.model_ctx()
        assert lib.test_tmp_size() == 6 * ctx * 32
        context = ctx if context is None else context
        assert ctx == context
        out = (ctypes.c_uint16 * 1024)()
        n = lib.test_generate(1, 32, 1, 0, out)
        assert list(out[:n]) == want
        assert keep
