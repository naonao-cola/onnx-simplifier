# Running the QDQ ResNet-18 graph on the phone

The simulator gives the per-kernel breakdown; the phone gives the number that matters. Both are
needed, and they are not comparable - see RESNET18_PERF.md.

## The skel is prebuilt

`qdq_net.py --skel` needs the **Hexagon SDK's `qaic` and FastRPC headers**, which are not on
this box (only the open-access toolchain under `~/.cache/hexagon-oa-19/Tools` is). A build of
the whole-graph program already exists, produced by the earlier session:

```
/mnt/data/cache/claude-work/hmxwork/tgnet_rn18/
  tg_hmx_rpc.so   client   blob.bin   input.bin   graph.h   k*.c   sim_profile.txt
```

`blob.bin` is the constants blob (11 MB) that the skel maps, so **a code change needs a rebuild
and therefore the SDK**; a measurement of the current build does not.

## Measuring the current build

```bash
S=239dbd8f; D=/data/local/tmp/rn18-baseline; O=/mnt/data/cache/claude-work/hmxwork/tgnet_rn18
A="adb -s $S"
$A shell "mkdir -p $D/case"
$A push $O/tg_hmx_rpc.so $O/client $D/
$A push $O/input.bin     $D/case/a.bin     # the network input
$A push $O/blob.bin      $D/case/b.bin     # the constants blob
$A push /mnt/data/cache/claude-work/hmxwork/resnet/ref.bin $D/case/ref.bin   # ORT CPU's output
$A shell "chmod 755 $D/client"
PHONE_LOCK_OWNER=<branch> ~/.cache/android-phone/phone-run $A shell "cd $D && \
  LD_LIBRARY_PATH=/vendor/lib64 ADSP_LIBRARY_PATH=$D timeout 300 ./client \
  'file:///tg_hmx_rpc.so?tg_hmx_rpc_skel_handle_invoke&_modver=1.0&_dom=cdsp' case 10; echo exit=\$?"
```

The client prints `codes power 0 ... hmx 0 vtcm 4194304`, the mismatch count against `ref.bin`,
and `us/inference` over the requested iterations. **`codes ... hmx 0` matters**: the HMX power
vote (`HAP_power_set_HMX`) is mandatory, and without it the first HMX tile op hangs the cDSP
until the phone reboots. `hmx 0` means the vote succeeded. Do not skip it because "the kernel
seemed fine last time".

Baseline recorded 2026-09-26 on this phone (SM8475 / taro, V69), 10 iterations:

```
rc 0 codes power 0 ctx 1946473852 hvx 0 hmx 0 vtcm 4194304 thread 0 heap 16934 KB; 0/25088 mismatches
22306.6 us/inference (10 iters) PASS
```

## Rules

- Every device command goes through the phone lock, and only for the device step - not for the
  build or the analysis. A holder file whose pid is dead means the lock is free.
- Keep to your own `/data/local/tmp/<branch>/`.
- `scripts/android/hmx_probe/run.sh health` after a run that touched HMX, if the build is
  available.
- A run that reports mismatches, or `codes` with a nonzero field, is not a measurement. Stop and
  read it.

## What a change needs

1. `qdq_net.py ... --sim` on the simulator first: it is ~3 minutes against ~4 on the phone, and
   it is the only way to get the per-call breakdown that says which kernel moved.
2. Then the phone, to confirm. Integer results transfer between the two; float ones do not (the
   simulator runs `.sf` as IEEE where V69 hardware computes qf32 - a requant exact on the
   simulator was 94% wrong on the phone).
3. If the two disagree on speed, trust the phone and treat the simulator's per-kernel shares as a
   guide to where to look, not as the result.
