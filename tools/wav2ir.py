#!/usr/bin/env python3
"""wav2ir.py - convert an impulse response WAV into a C header for embedding.

Usage:
    python tools/wav2ir.py input.wav [-o ir_data.h] [--rate 48000]
                           [--max-seconds 3.0] [--fade-ms 20]
                           [--trim-db -60] [--stereo]
                           [--gain-db 0] [--no-normalize] [--name IrData]

- Accepts PCM 8/16/24/32-bit and float32/float64 WAV, mono or multichannel
  (channels are averaged to mono, or kept as left/right with --stereo).
- Resamples to the target rate (windowed-sinc) if needed.
- --trim-db X drops the leading samples quieter than X dB below the peak
  (pre-delay adds latency; off by default).
- Truncates to --max-seconds; a truncated tail gets a --fade-ms raised-
  cosine fade-out so it does not stop abruptly.
- By default normalizes to unit energy (sum h^2 = 1, i.e. ~unity loudness
  through the convolver; with --stereo the two channels share one factor),
  then applies --gain-db.

The output header defines kIrData[] (or kIrDataL[] / kIrDataR[] and
IR_DATA_STEREO with --stereo, kIrData then being the left channel),
kIrDataLen and kIrDataSampleRate. --name Foo renames them (kFoo[],
kFooLen, kFooSampleRate, kFooL[] / kFooR[], FOO_STEREO) so several IRs can
be embedded side by side. Drop it next to
examples/convolver/convolver.cpp as `ir_data.h`; that Makefile picks it up
automatically.
"""

import argparse
import math
import re
import struct
import sys


def read_wav(path):
    """Minimal RIFF parser: returns (channels, sample_rate), channels being
    a list of per-channel float lists."""
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 12 or data[0:4] != b"RIFF" or data[8:12] != b"WAVE":
        sys.exit(f"error: {path} is not a RIFF/WAVE file")

    fmt = None
    raw = None
    pos = 12
    while pos + 8 <= len(data):
        cid = data[pos : pos + 4]
        (size,) = struct.unpack_from("<I", data, pos + 4)
        body = data[pos + 8 : pos + 8 + size]
        if cid == b"fmt ":
            fmt = body
        elif cid == b"data":
            raw = body
        pos += 8 + size + (size & 1)  # chunks are word-aligned

    if fmt is None or raw is None:
        sys.exit("error: missing fmt/data chunk")

    tag, nch, rate, _, _, bits = struct.unpack_from("<HHIIHH", fmt, 0)
    if tag == 0xFFFE:  # WAVE_FORMAT_EXTENSIBLE: real tag in the GUID
        (tag,) = struct.unpack_from("<H", fmt, 24)
    if nch < 1 or bits % 8:
        sys.exit(f"error: unsupported WAV layout ({nch} ch, {bits} bits)")

    width = bits // 8
    nframes = len(raw) // (nch * width)
    count = nframes * nch
    if tag == 1 and bits == 8:  # unsigned
        vals = [(b - 128) / 128.0 for b in raw[:count]]
    elif tag == 1 and bits == 16:
        vals = [v / 32768.0 for v in struct.unpack_from(f"<{count}h", raw)]
    elif tag == 1 and bits == 24:
        vals = [
            int.from_bytes(raw[3 * i : 3 * i + 3], "little", signed=True)
            / 8388608.0
            for i in range(count)
        ]
    elif tag == 1 and bits == 32:
        vals = [v / 2147483648.0 for v in struct.unpack_from(f"<{count}i", raw)]
    elif tag == 3 and bits == 32:
        vals = list(struct.unpack_from(f"<{count}f", raw))
    elif tag == 3 and bits == 64:
        vals = list(struct.unpack_from(f"<{count}d", raw))
    else:
        sys.exit(f"error: unsupported WAV format (tag={tag}, bits={bits})")

    channels = [vals[c::nch] for c in range(nch)]
    return channels, rate


def resample(x, rate_in, rate_out, taps_per_side=32):
    """Windowed-sinc resampler (Blackman window), good enough for IRs."""
    if rate_in == rate_out:
        return x
    ratio = rate_out / rate_in
    n_out = int(len(x) * ratio)
    cutoff = 0.95 * min(1.0, ratio)  # in units of input Nyquist
    y = []
    for n in range(n_out):
        center = n / ratio
        i0 = int(math.floor(center)) - taps_per_side + 1
        acc = 0.0
        for i in range(i0, i0 + 2 * taps_per_side):
            if 0 <= i < len(x):
                t = center - i
                # sinc low-pass * Blackman window
                s = cutoff * (math.sin(math.pi * cutoff * t) / (math.pi * cutoff * t) if t != 0.0 else 1.0)
                w = t / taps_per_side
                if abs(w) < 1.0:
                    win = 0.42 + 0.5 * math.cos(math.pi * w) + 0.08 * math.cos(2 * math.pi * w)
                else:
                    win = 0.0
                acc += x[i] * s * win
        y.append(acc)
    return y


def write_array(f, name, x):
    f.write(f"static const float {name}[{len(x)}] = {{\n")
    for i in range(0, len(x), 6):
        row = ", ".join(f"{v:.8e}f" for v in x[i : i + 6])
        f.write(f"    {row},\n")
    f.write("};\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("input")
    ap.add_argument("-o", "--output", default="ir_data.h")
    ap.add_argument("--rate", type=int, default=48000)
    ap.add_argument("--max-seconds", type=float, default=3.0,
                    help="truncate the IR to this length (engine max)")
    ap.add_argument("--fade-ms", type=float, default=20.0,
                    help="fade-out of a truncated tail (0: hard cut)")
    ap.add_argument("--trim-db", type=float, default=None,
                    help="drop leading samples below this level relative "
                         "to the peak, e.g. -60")
    ap.add_argument("--stereo", action="store_true",
                    help="keep two channels (left/right IRs)")
    ap.add_argument("--gain-db", type=float, default=0.0)
    ap.add_argument("--no-normalize", action="store_true",
                    help="skip unit-energy normalization")
    ap.add_argument("--name", default="IrData",
                    help="C identifier stem of the generated names")
    args = ap.parse_args()

    chans, rate = read_wav(args.input)
    print(f"in : {args.input}: {len(chans[0])} samples x {len(chans)} ch "
          f"@ {rate} Hz ({len(chans[0]) / rate:.3f} s)")
    if args.stereo:
        if len(chans) < 2:
            sys.exit("error: --stereo needs a file with two channels")
        chans = chans[:2]
    else:
        n = len(chans)
        chans = [[sum(v) / n for v in zip(*chans)]]

    if rate != args.rate:
        print(f"     resampling {rate} -> {args.rate} Hz ...")
        chans = [resample(x, rate, args.rate) for x in chans]

    if args.trim_db is not None:
        peak = max(abs(v) for x in chans for v in x)
        thr = peak * 10.0 ** (args.trim_db / 20.0)
        start = min(
            next((i for i, v in enumerate(x) if abs(v) >= thr), len(x))
            for x in chans
        )
        if start:
            print(f"     trimming {start} leading samples "
                  f"({1000.0 * start / args.rate:.2f} ms below "
                  f"{args.trim_db:g} dB)")
            chans = [x[start:] for x in chans]

    max_len = int(args.max_seconds * args.rate)
    if len(chans[0]) > max_len:
        print(f"     truncating to {args.max_seconds} s ({max_len} samples)")
        chans = [x[:max_len] for x in chans]
        nfade = min(max_len, int(args.fade_ms * args.rate / 1000.0))
        if nfade > 0:
            print(f"     {args.fade_ms:g} ms fade-out")
            for x in chans:
                for i in range(nfade):
                    g = 0.5 + 0.5 * math.cos(math.pi * (i + 1) / nfade)
                    x[max_len - nfade + i] *= g

    if not args.no_normalize:
        energy = math.sqrt(sum(v * v for x in chans for v in x) / len(chans))
        if energy > 0.0:
            chans = [[v / energy for v in x] for x in chans]
    g = 10.0 ** (args.gain_db / 20.0)
    if g != 1.0:
        chans = [[v * g for v in x] for x in chans]

    name = args.name
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        sys.exit(f"error: --name {name!r} is not a C identifier")
    macro = ("IR_DATA" if name == "IrData"
             else re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).upper())
    n = len(chans[0])
    with open(args.output, "w") as f:
        f.write(f"// Generated by wav2ir.py from {args.input}\n")
        f.write(f"// {n} samples x {len(chans)} ch @ {args.rate} Hz"
                f" ({n / args.rate:.3f} s)\n")
        f.write("#pragma once\n\n")
        f.write(f"static const unsigned k{name}Len = {n}u;\n")
        f.write(f"static const float k{name}SampleRate = {args.rate}.0f;\n")
        if args.stereo:
            f.write(f"#define {macro}_STEREO 1\n")
            write_array(f, f"k{name}L", chans[0])
            write_array(f, f"k{name}R", chans[1])
            f.write(f"static const float* const k{name} = k{name}L;\n")
        else:
            write_array(f, f"k{name}", chans[0])
    print(f"out: {args.output}: {n} samples x {len(chans)} ch @ {args.rate} Hz")


if __name__ == "__main__":
    main()
