/* Flat, position-specific binary: everything at ORG (set with --defsym ORG=0x...). */
SECTIONS
{
    . = ORG;
    .text : {
        *(.text .text.*)
        *(.rodata .rodata.*)
        *(.data .data.*)
        *(.bss .bss.* COMMON)
    }
    /DISCARD/ : { *(.comment) *(.note*) *(.debug*) }
}
