#!/usr/bin/env python3
"""Build a hot-expert sidecar for the AUTOGRID/TAARDIS MoE cache.

Reads a base GGUF and an expert-hit profile, then writes a small adapter-style
GGUF holding the top-K experts of every routed-expert bank, plus per-layer
hot/cold masks.  The sidecar is what the fork's `build_moe_ffn_split` will pin
on the GPU while the base bank stays on the CPU; the split is exact because the
same normalized weights are applied to both halves (see SERVING-EXTRAS-PLAN.md).

Format (sidecar.version 1):
  metadata   general.type = "adapter", adapter.type = "taardis-lora",
             adapter.lora.alpha = 1.0, sidecar.kind = "moe-hot-cache"
  tensors    blk.N.ffn_{gate,up,down,gate_up}_exps.hot   (raw hot expert blocks,
                                                          same ggml type as base)
             blk.N.ffn_hot_mask  F32[n_expert]  (1.0 for hot)
             blk.N.ffn_cold_mask F32[n_expert]  (1.0 for cold)

Usage:
  python make_hot_sidecar.py --model olmoe-q1g128.gguf \
      --profile ../placement-sweep-20260927/expert-hits.json \
      --hot 16 --out olmoe-hot16.sidecar.gguf
  python make_hot_sidecar.py --model X.gguf --profile p.json --hot 16 --check
"""
from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from pathlib import Path

ALIGN = 32
GGUF_MAGIC = b"GGUF"
EXPERT_RE = re.compile(
    r"^blk\.(\d+)\.ffn_(gate_up|gate|up|down)_exps\.weight$")

# gguf metadata value types
T_U32, T_I32, T_F32, T_STRING = 4, 5, 6, 8

# ggml tensor element types (NOT the metadata types above)
GGML_T_F32, GGML_T_I32 = 0, 26


# ------------------------------------------------------------------ gguf read

def _read_str(f) -> str:
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode("utf-8", "replace")


def _skip_value(f, vtype: int) -> None:
    scalar = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
              10: 8, 11: 8, 12: 8}
    if vtype in scalar:
        f.seek(scalar[vtype], 1)
    elif vtype == T_STRING:
        f.seek(struct.unpack("<Q", f.read(8))[0], 1)
    elif vtype == 9:  # array
        etype = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        if etype == T_STRING:
            for _ in range(n):
                f.seek(struct.unpack("<Q", f.read(8))[0], 1)
        else:
            f.seek(n * scalar.get(etype, 1), 1)
    else:
        raise SystemExit(f"unknown gguf value type {vtype}")


def read_gguf_index(path: Path):
    """-> (metadata {key: (type, value)}, tensors {name: [dims, type, offset]},
    data_start)."""
    with open(path, "rb") as f:
        if f.read(4) != GGUF_MAGIC:
            raise SystemExit(f"{path}: not a GGUF file")
        version, n_tensors, n_kv = struct.unpack("<IQQ", f.read(20))
        meta = {}
        for _ in range(n_kv):
            key = _read_str(f)
            vtype = struct.unpack("<I", f.read(4))[0]
            if vtype == T_STRING:
                val = _read_str(f)
            elif vtype == T_U32:
                val = struct.unpack("<I", f.read(4))[0]
            elif vtype == T_F32:
                val = struct.unpack("<f", f.read(4))[0]
            else:
                _skip_value(f, vtype)
                val = None
            meta[key] = (vtype, val)
        tensors = {}
        for _ in range(n_tensors):
            name = _read_str(f)
            n_dims = struct.unpack("<I", f.read(4))[0]
            dims = list(struct.unpack(f"<{n_dims}Q", f.read(8 * n_dims)))
            ttype = struct.unpack("<I", f.read(4))[0]
            off = struct.unpack("<Q", f.read(8))[0]
            tensors[name] = [dims, ttype, off]
        header_end = f.tell()
    align = int(meta.get("general.alignment", (T_U32, ALIGN))[1] or ALIGN)
    data_start = (header_end + align - 1) // align * align
    return meta, tensors, data_start


# ----------------------------------------------------------------- gguf write

def _w_str(b: bytearray, s: str) -> None:
    raw = s.encode()
    b += struct.pack("<Q", len(raw)) + raw


def _w_kv_str(b: bytearray, key: str, val: str) -> None:
    _w_str(b, key)
    b += struct.pack("<I", T_STRING)
    _w_str(b, val)


def _w_kv_f32(b: bytearray, key: str, val: float) -> None:
    _w_str(b, key)
    b += struct.pack("<I", T_F32) + struct.pack("<f", val)


def _w_kv_u32(b: bytearray, key: str, val: int) -> None:
    _w_str(b, key)
    b += struct.pack("<I", T_U32) + struct.pack("<I", val)


def write_sidecar(out: Path, arch: str, hot_size: int, tensors: list[tuple[str, list[int], int]], blobs: list[bytes]) -> None:
    kv_items = [
        ("general.architecture", "str", arch),
        ("general.type", "str", "adapter"),
        ("adapter.type", "str", "taardis-lora"),
        ("adapter.lora.alpha", "f32", 1.0),
        ("sidecar.kind", "str", "moe-hot-cache"),
        ("sidecar.version", "u32", 1),
        ("sidecar.hot", "u32", hot_size),
        ("general.alignment", "u32", ALIGN),
    ]
    kv = bytearray()
    for key, kind, val in kv_items:
        if kind == "str":
            _w_kv_str(kv, key, val)
        elif kind == "f32":
            _w_kv_f32(kv, key, val)
        else:
            _w_kv_u32(kv, key, val)

    head = bytearray()
    head += GGUF_MAGIC
    head += struct.pack("<IQQ", 3, len(tensors), len(kv_items))
    head += kv
    offset = 0
    for (name, dims, ttype), blob in zip(tensors, blobs):
        _w_str(head, name)
        head += struct.pack("<I", len(dims))
        for d in dims:
            head += struct.pack("<Q", d)
        head += struct.pack("<I", ttype)
        head += struct.pack("<Q", offset)
        offset += (len(blob) + ALIGN - 1) // ALIGN * ALIGN

    pad = (-len(head)) % ALIGN
    with open(out, "wb") as f:
        f.write(head)
        f.write(b"\0" * pad)
        for blob in blobs:
            f.write(blob)
            f.write(b"\0" * ((-len(blob)) % ALIGN))


# ---------------------------------------------------------------------- main

def top_k(counts: list[int], k: int) -> list[int]:
    order = sorted(range(len(counts)), key=lambda i: -counts[i])
    return sorted(order[:k])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="base GGUF")
    ap.add_argument("--profile", required=True, help="expert-hits.json")
    ap.add_argument("--hot", type=int, required=True,
                    help="experts to keep per layer (uniform K)")
    ap.add_argument("--out", help="sidecar output path")
    ap.add_argument("--check", action="store_true",
                    help="with --out: verify an existing sidecar against the base")
    a = ap.parse_args(argv)

    model = Path(a.model)
    model_size = model.stat().st_size
    meta, tensors, data_start = read_gguf_index(model)
    arch = meta.get("general.architecture", (0, "?"))[1]
    prof = json.loads(Path(a.profile).read_text())
    counts = prof["counts"]                      # [layers][experts]
    n_experts = len(counts[0])

    banks = [(name, m) for name, m in sorted(tensors.items())
             if EXPERT_RE.match(name)]
    if not banks:
        raise SystemExit("no routed-expert tensors found")

    # group by layer so every projection gets the same hot set
    hot_by_layer = {li: top_k(counts[li], a.hot) for li in range(len(counts))}
    if a.check:
        return check_sidecar(Path(a.out), model, tensors, data_start, hot_by_layer)

    out_tensors, out_blobs = [], []
    total_hot = 0
    rel_sizes = None
    with open(model, "rb") as f:
        for name, m in banks:
            dims, ttype, off = m
            layer = int(EXPERT_RE.match(name).group(1))
            n_exp = dims[2]
            if n_exp != n_experts:
                raise SystemExit(f"{name}: {n_exp} experts, profile has {n_experts}")
            size = tensor_size(tensors, name, data_start, model_size)
            if size % n_exp:
                raise SystemExit(f"{name}: bytes not divisible by experts")
            stride = size // n_exp
            hot = hot_by_layer[layer]
            blob = bytearray()
            for e in hot:
                f.seek(data_start + off + e * stride)
                blob += f.read(stride)
            base = name[: -len(".weight")] + ".hot"
            out_tensors.append((base, [dims[0], dims[1], len(hot)], ttype))
            out_blobs.append(bytes(blob))
            total_hot += len(blob)
            if rel_sizes is None:
                rel_sizes = size
            # masks
        for li in range(len(counts)):
            hot = hot_by_layer[li]
            hot_set = set(hot)
            local = {e: j for j, e in enumerate(hot)}
            hm = [1.0 if e in hot_set else 0.0 for e in range(n_experts)]
            cm = [0.0 if e in hot_set else 1.0 for e in range(n_experts)]
            mv = [local.get(e, 0) for e in range(n_experts)]
            out_tensors.append((f"blk.{li}.ffn_hot_map", [n_experts], GGML_T_I32))
            out_blobs.append(struct.pack(f"<{n_experts}i", *mv))
            for suffix, vec in (("hot_mask", hm), ("cold_mask", cm)):
                nm = f"blk.{li}.ffn_{suffix}"
                out_tensors.append((nm, [n_experts], GGML_T_F32))
                out_blobs.append(struct.pack(f"<{n_experts}f", *vec))

    out = Path(a.out)
    write_sidecar(out, arch, a.hot, out_tensors, out_blobs)

    # ---- verify by reading back and comparing a sample of hot blocks ----
    meta2, tensors2, ds2 = read_gguf_index(out)
    bad = 0
    with open(model, "rb") as fin, open(out, "rb") as fout:
        for i, (name, m) in enumerate(banks):
            dims, ttype, off = m
            layer = int(EXPERT_RE.match(name).group(1))
            n_exp = dims[2]
            stride = tensor_size(tensors, name, data_start, model_size) // n_exp
            base_size = tensor_size(tensors2, name[: -len(".weight")] + ".hot",
                                    ds2, out.stat().st_size)
            hot = hot_by_layer[layer]
            if base_size != stride * len(hot):
                bad += 1
                continue
            for j in (0, len(hot) - 1):        # spot-check first + last
                fin.seek(data_start + off + hot[j] * stride)
                fout.seek(ds2 + tensors2[name[: -len(".weight")] + ".hot"][2] + j * stride)
                if fin.read(stride) != fout.read(stride):
                    bad += 1
                    break
    print(f"sidecar     {out}")
    print(f"arch        {arch} | experts {n_experts} | hot {a.hot}/layer")
    print(f"tensors     {len(out_tensors)} ({len(banks)} banks + "
          f"{3 * len(counts)} map/masks)")
    print(f"hot bytes   {total_hot / 2**20:.1f} MiB "
          f"({100 * a.hot / n_experts:.0f}% of experts)")
    print(f"verify      {'OK' if bad == 0 else f'{bad} MISMATCHES'}")
    return 0 if bad == 0 else 1


def tensor_size(tensors: dict, name: str, data_start: int, file_size: int) -> int:
    """Exact tensor byte size from the offset table (alignment padding is in
    the gap, <= ALIGN bytes — irrelevant for slicing because offsets are
    aligned).  The last tensor in the file needs data_start + file_size."""
    ordered = sorted(tensors.items(), key=lambda kv: kv[1][2])
    for i, (n, m) in enumerate(ordered):
        if n != name:
            continue
        if i + 1 < len(ordered):
            return ordered[i + 1][1][2] - m[2]
        return file_size - data_start - m[2]
    raise KeyError(name)


def check_sidecar(out: Path, model: Path, tensors, data_start, hot_by_layer) -> int:
    meta2, tensors2, ds2 = read_gguf_index(out)
    model_size = Path(model).stat().st_size
    bad = 0
    with open(model, "rb") as fin, open(out, "rb") as fout:
        for name, m in sorted(tensors.items()):
            if not EXPERT_RE.match(name):
                continue
            dims, ttype, off = m
            layer = int(EXPERT_RE.match(name).group(1))
            hot = hot_by_layer[layer]
            hot_name = name[: -len(".weight")] + ".hot"
            if hot_name not in tensors2:
                bad += 1
                continue
            stride = tensor_size(tensors, name, data_start, model_size) // dims[2]
            for j, e in enumerate(hot):
                fin.seek(data_start + off + e * stride)
                fout.seek(ds2 + tensors2[hot_name][2] + j * stride)
                if fin.read(stride) != fout.read(stride):
                    bad += 1
                    break
    print(f"check {out}: {'OK' if bad == 0 else f'{bad} MISMATCHES'}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
