"""Build TensorRT engines on this machine from ONNX graphs produced on the GB10 (same graphs, local engines)."""
import sys, os, json, glob
sys.path.insert(0, sys.argv[1])
from fusion3d.deploy import build_engine
d, tag = sys.argv[2], sys.argv[3]
info = {}
for part in ['A', 'B']:
    q = os.path.join(d, f'{tag}_{part}.onnx')
    if os.path.exists(q):
        info[f'{part}_build_s'] = build_engine(q, os.path.join(d, f'{tag}_{part}.engine'))
print('BUILT', tag, json.dumps(info))
