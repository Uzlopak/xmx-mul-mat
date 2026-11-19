#!/usr/bin/env python3
"""Check the immutable upstream snapshot; optionally require an unchanged kernel."""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--strict", action="store_true", help="Also fail if the working kernel differs from the extracted original")
args = parser.parse_args()
root = Path(__file__).resolve().parent.parent
metadata = json.loads((root / "UPSTREAM.json").read_text())
original = (root / "third_party/llama.cpp/xmx-kernel.original.inc").read_bytes()
if hashlib.sha256(original).hexdigest() != metadata["extracted_body_sha256"]:
    raise SystemExit("FAIL: the immutable upstream snapshot does not match UPSTREAM.json")
working = (root / "include/xmx/kernel.hpp").read_bytes()
start = working.index(b"//\n// Multi-column mul_mat_vec for reordered K-quant weights on the XMX engines")
end_marker = b"#endif // GGML_SYCL_MMVQ_HAS_XMX\n"
end = working.index(end_marker, start) + len(end_marker)
unchanged = working[start:end] == original
print("Upstream snapshot: verified")
print("Working kernel:", "unchanged" if unchanged else "modified for development")
if args.strict and not unchanged:
    raise SystemExit(1)
