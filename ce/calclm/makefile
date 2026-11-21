# ----------------------------
# CALCLM: CE C toolchain program (~/CEdev). Build:  PATH=$HOME/CEdev/bin:$PATH make
# The inference core is compiled once by tools/build_core.sh into ../build/calclm/core.o,
# the same object the CEmu tests (ce/tests/test_calclm*.py) link and run.
# ----------------------------

NAME = CALCLM
DESCRIPTION = "CalcLM: writes TI-BASIC"
COMPRESSED = YES
ARCHIVED = YES
HAS_PRINTF = NO

CFLAGS = -Wall -Wextra -Oz -I$(CURDIR)/core -I$(CURDIR)/../../tracker
CXXFLAGS = -Wall -Wextra -Oz

# SHAPE=w builds for the wide model W (core/clm_shape_w.h) into bin/w, obj/w and
# ../build/calclm-w; the default is v2.2's shape (core/clm_shape.h).
SHAPE ?=
ifneq ($(SHAPE),)
BINDIR = bin/$(SHAPE)
OBJDIR = obj/$(SHAPE)
CFLAGS += -include $(CURDIR)/core/clm_shape_$(SHAPE).h
endif

CORE_O = ../build/calclm$(if $(SHAPE),-$(SHAPE))/core.o
EXTRA_LIBS = $(CORE_O)
DEPS = $(CORE_O)

# ----------------------------

include $(shell cedev-config --makefile)

$(CORE_O): $(wildcard core/*.c core/*.h core/ez80/*.s core/ez80/*.inc ../../tracker/tracker.[ch] ../../tracker/trk_tables.h) tools/build_core.sh tools/gen_kernels.py ../kernel/gen_tmv.py
	$(if $(SHAPE),CLM_SHAPE=core/clm_shape_$(SHAPE).h CALCLM_OUT=$(CURDIR)/../build/calclm-$(SHAPE) CALCLM_BIN=$(CURDIR)/bin/$(SHAPE)) tools/build_core.sh
