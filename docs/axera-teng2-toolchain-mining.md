# Does the toolchain already have a shortcut past manual `teng2` decode? Confirmed no, plus one untested lead and one new terminology find

Before spending more hours on byte-level fault injection to reverse-engineer
AX650's `teng2` compute microprogram, this checks whether the Pulsar2
toolchain or AXCL SDK already contains an undocumented disassembler, verbose
MCode dump flag, or embedded ISA reference that would shortcut the whole
problem. `scripts/axera/README.md`'s firmware-string-mining section already
found real internal terminology this way once; this pushes that same kind of
hunt further and closes it out more systematically.

## What was already known, re-verified rather than re-discovered

`README.md`'s "The AX650N card's own firmware" section already found the
answer to the headline question, and it still holds: Pulsar2's own backend
simulator (`/opt/pulsar2/backend/ax650npu/ax650npu_cmodel.so` inside the
Docker image) exports a genuine, undocumented-elsewhere mcode assembler and
disassembler as plain C symbols -- `mcode_new`, `mcode_dump`, `mcode_size`,
`mcode_disassemble`, `assembler_eu`, `assembler_ctrl`, `disassembler_eu` --
exactly the tool that would turn this whole investigation's differential
byte-diffing into direct disassembly. It is gated by a Sentinel LDK hardware/
software dongle check fired from the library's own load-time constructor
(`dlopen()` alone fails with `Sentinel LDK Protection System: Sentinel key
not found (H0007)`). This is a legitimate commercial licensing control on
Axera's own tooling, not pursued here, matching the prior doc's own
conclusion.

**New in this pass**: Pulsar2's own CLI (`/opt/pulsar2/axnn/*.py`) is itself
Pyarmor-obfuscated -- `#! /usr/bin/env python3` followed by a Pyarmor 9.1.3
runtime-decrypted bytecode blob, not plain source, despite the `.py`
extension and the plain-text tracebacks Pulsar2 prints (those come from
`<frozen ...>` module names baked into the encrypted bytecode's own
exception metadata, not from readable source on disk). This is a second,
separate commercial protection layer -- also not pursued, for the same
reason as the Sentinel dongle: it is a real, deliberate protection
mechanism on proprietary software, not a bug to route around.

## New: the hidden `--debug.*` flag namespace is closed and small

`pulsar2 build --help` does not list any `--debug.*` option, yet
`scripts/axera/pulsar2_docker.py` already relies on `--debug.dump_frontend_graph`
existing and working. Testing confirms *why* it's undocumented rather than
broken: passing it alone (no `--input`) proceeds past argument parsing into
the build pipeline and fails on `IsADirectoryError` for the default input
path `.` -- i.e. it is accepted. Passing a bogus flag,
`--debug.xxxxxxx`, is rejected immediately at the top-level argparse layer:
`main.py: error: unrecognized arguments: --debug.xxxxxxx`. So there is a
real, closed, enumerable set of hidden debug flags checked at parse time,
just excluded from `--help` output -- and the accept/reject distinction is a
fast, reliable, black-box oracle (a few seconds per candidate, no full build
needed) for finding the rest of that set without touching the obfuscated
source.

30 plausible sibling names were probed this way (`dump_mcode`, `dump_neu`,
`dump_wbt`, `dump_ir`, `dump_schedule`, `dump_optimized_graph`,
`dump_backend_graph`, `dump_quant_graph`, `dump_cmodel`, `dump_npu_graph`,
`dump_cmodel_trace`, `enable_disasm`, `disassemble`, `verbose`, `dump_trace`,
`dump_debug_info`, `dump_intermediate`, `dump_engine_trace`,
`keep_temp_files`, `dump_all`, `dump_neu_graph`, `dump_compiled_graph`,
`dump_mcode_text`, `dump_disasm`, `mcode_disasm`, `dump_command_queue`,
`dump_cmdq`, `dump_eu_trace`, `enable_verbose_log`, `log_level`,
`dump_graph`). **All 30 were rejected.** `--debug.dump_frontend_graph`
remains the only accepted flag in this namespace found so far. This does not
prove the set has exactly one member -- only that these 30 specific guesses
aren't it -- but it rules out every name a person would plausibly try next,
so it's a real, if incomplete, negative result.

`pulsar2 run --help`, `pulsar2 llm_build[2] --help`, and `pulsar2 version`
add nothing beyond what `pulsar2_docker.py` already uses (`--enable_perlayer_output`
on `run` is the closest thing to a trace flag there, and it dumps
per-layer *tensor values*, not instruction-level detail -- already a known,
ordinary capability, not a new lead).

## New: `TENG EU` is a real, named hardware block -- confirming this project's own terminology, not inventing it

`README.md`'s firmware dig already `nm`'d the (unstripped) `ax_npu.ko`
kernel driver's symbol table and found five identically-compiled
queue-setter functions -- `npu_dma_set_queue`, `npu_mau_set_queue`,
`npu_potato_set_queue`, `npu_sdma_set_queue`, `npu_warp_set_queue`. This
pass went past symbol *names* to the driver's embedded log/error *format
strings* (`strings` on the extracted `.ko`, not just `nm`), and found seven
distinct execution-unit types named in `[NPU][%s][%s %d]: <NAME> EU[%d]:
...` error strings: `CONV`, `CV`, `DMA`, `MAU`, `SDMA`, `TENG`, `WARP`.

**`TENG` is the first direct hardware-level confirmation of this project's
own "teng2" naming** (taken from `trace.json`'s engine names throughout this
project's profiling work) -- it is a real execution-unit type with its own
CRC/AXI/retrigger error strings (`TENG EU[%d]: Cmd Crc Error`,
`Rdma0 Ch0 Crc Error`, `Retrigger error`, ...), not a project-invented label.
Neither `TENG` nor `CONV` nor `CV` nor `WARP` has a matching `npu_*_set_queue`
symbol the way `DMA`/`MAU`/`POTATO`/`SDMA` do -- their queue setup is
evidently handled through a different code path (or fully static/always-on),
worth noting for anyone extending the driver-symbol table further. This is
useful confirmatory terminology, not a disassembly shortcut: the format
strings are generic status/error reporting (`%d`/`%x` counters and error
codes), not instruction mnemonics or an opcode table.

A broad filesystem search of the whole Docker image for anything else
disassembler/ISA-shaped (`*disasm*`, `*opcode*`, `*isa_table*`,
`*instr_set*`) found nothing NPU-related -- every hit was a generic,
unrelated tool already present for other reasons (Python's own bytecode
`opcode` module, Perl's `Opcode` module, GNU binutils' `libopcodes` for
x86/ARM). There is no second, unprotected copy of the mcode disassembler
anywhere in the image.

## A real, legitimate, untested lead: AXCL's own documented trace-log level

`/usr/bin/axcl/axcl.json` (the AXCL host runtime's own config file, not
reverse-engineered -- read directly, it ships with an explanatory comment)
has:

```json
"log": {
  "host":   {"level": 2, "//   ": "0: trace, 1: debug, 2: info, 3: warn, 4: error, 5: critical, 6: off"},
  "device": {"level": 2, "//   ": "0: trace, 1: debug, 2: info, 3: warn, 4: error, 5: critical, 6: off"}
}
```

This is a genuine, documented, first-party configuration option -- not a
protection bypass of any kind -- and it was **not tested in this pass**,
since this task's scope excluded device runs (no AXCL device is attached to
this host; only the `axcl-vm` LXD VM this project's device work already uses
has one). Setting `device.level` to `0` (trace) before running a real
`.axmodel` through `axcl_run_model` on `axcl-vm` could plausibly surface
per-instruction execution detail at runtime -- a dynamic instruction trace,
which would be at least as useful as a static disassembler for confirming
what a `teng2` verb actually *does*, and arguably more useful since it would
show real operand values in context rather than just a decoded mnemonic.
This is the single most promising concrete next step from this whole pass,
and it is cheap to try: no device runs were made here, so it is completely
untested, not tried-and-failed.

`axcl_run_model --help`, and a `strings` pass over `axcl_run_model`,
`axcl-smi`, and `axcl_sample_sys` for both flag-shaped and `AXCL_*`
environment-variable-shaped tokens, found nothing beyond the already-documented
CLI options (`--verify`, `-x/--api`, etc.) and ordinary `AXCL_ENGINE_*`/
`AXCL_ERR_*` API/error-code names -- no hidden trace flag or env var exists
on the *host* binaries; `axcl.json`'s log level is the only lever found.

## Bottom line

No shortcut exists that doesn't involve circumventing a real commercial
protection mechanism (the Sentinel LDK dongle, or Pyarmor's bytecode
encryption) -- confirmed by re-verifying the prior finding and extending the
search across the CLI flag surface, the whole Docker image's filesystem, and
the AXCL host binaries, all empty-handed for anything new and unprotected.
Manual, differential byte-level decode -- what every `teng2`/segment-2 doc in
this project has been doing -- remains the only available path *unless*
`axcl.json`'s trace-level device log is tried and turns out to expose real
instruction-level detail. That one lever is genuine, cheap, and completely
untested; it is the concrete thing to try next, not another round of static
searching.

## Reproduction

- CLI flag probe: `docker run --rm pulsar2:7.0-lite pulsar2 build --debug.<name>`;
  `unrecognized arguments` in the output means rejected, anything else
  (typically an `IsADirectoryError` traceback, since no `--input` is given)
  means accepted.
- Firmware extraction: `dd`/direct byte-range read of
  `/usr/lib/firmware/axcl/ax650_card.pac` at the rootfs partition's
  documented `(offset=21578033, size=134217728)` (see `README.md`'s firmware
  section for the full partition table), then `debugfs -R "dump /soc/ko/ax_npu.ko <out>"`
  on the extracted image, then `strings -n 3 <out> | grep -i teng` (or any
  other execution-unit name) to reproduce the `TENG EU` finding.
- AXCL binaries: present on this host directly at `/usr/bin/axcl/` (package
  `axclhost`), no VM needed for the static checks above; `axcl-smi`/
  `axcl_run_model` report no local device (`/dev/axcl_host` does not exist on
  this host), so any dynamic/device-facing check needs `axcl-vm`.
