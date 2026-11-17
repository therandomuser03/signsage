#!/usr/bin/env python3
"""Compute world-space rest pose of key bones from the GLB (no three.js needed)."""
import json, struct, sys
import numpy as np

def read_glb(path):
    with open(path, 'rb') as f:
        magic, version, length = struct.unpack('<III', f.read(12))
        chunks = []
        while f.tell() < length:
            clen, ctype = struct.unpack('<II', f.read(8))
            chunks.append((ctype, f.read(clen)))
    return json.loads(next(c for t, c in chunks if t == 0x4E4F534A))

def quat_to_mat(q):
    x, y, z, w = q
    n = x*x + y*y + z*z + w*w
    if n == 0: return np.eye(4)
    s = 2.0 / n
    xx, yy, zz = x*x*s, y*y*s, z*z*s
    xy, xz, yz = x*y*s, x*z*s, y*z*s
    wx, wy, wz = w*x*s, w*y*s, w*z*s
    m = np.array([
        [1-(yy+zz), xy-wz, xz+wy],
        [xy+wz, 1-(xx+zz), yz-wx],
        [xz-wy, yz+wx, 1-(xx+yy)],
    ])
    return m

def trs_to_mat(t, r, s):
    m = np.eye(4)
    m[:3,:3] = quat_to_mat(r) * np.array(s)
    m[:3,3] = t
    return m

def main(path):
    g = read_glb(path)
    nodes = g['nodes']
    parent = {}
    for i, n in enumerate(nodes):
        for c in n.get('children', []):
            parent[c] = i
    world = {}
    def compute(i):
        if i in world: return world[i]
        n = nodes[i]
        local = trs_to_mat(n.get('translation', [0,0,0]), n.get('rotation', [0,0,0,1]), n.get('scale', [1,1,1]))
        world[i] = (world[parent[i]] if i in parent else np.eye(4)) @ local
        return world[i]
    for i in range(len(nodes)): compute(i)
    by_name = {n.get('name', f'#{i}'): i for i, n in enumerate(nodes)}

    keys = ['root','spine05','spine04','spine03','spine02','spine01','clavicle.L','shoulder01.L','upperarm01.L','upperarm02.L','lowerarm01.L','lowerarm02.L','wrist.L',
            'clavicle.R','shoulder01.R','upperarm01.R','upperarm02.R','lowerarm01.R','lowerarm02.R','wrist.R','neck01','neck03','head',
            'metacarpal1.L','metacarpal2.L','metacarpal3.L','metacarpal4.L','finger1-1.L','finger2-1.L','finger3-1.L','finger4-1.L','finger5-1.L',
            'finger2-3.L','finger3-3.L','finger1-3.L','finger5-3.L','metacarpal1.R','metacarpal2.R','metacarpal3.R','metacarpal4.R','finger1-1.R','finger2-1.R','finger3-1.R','finger4-1.R','finger5-1.R','finger2-3.R','finger3-3.R','finger1-3.R','finger5-3.R']
    print(f"{'bone':16} {'world pos':34} {'local Y in world':34} {'local X in world'}")
    for k in keys:
        i = by_name.get(k)
        if i is None:
            print(k, 'MISSING'); continue
        w = world[i]
        p = w[:3,3]
        y = w[:3,1]; x = w[:3,0]; z = w[:3,2]
        print(f"{k:16} [{p[0]: .4f},{p[1]: .4f},{p[2]: .4f}]   [{y[0]: .3f},{y[1]: .3f},{y[2]: .3f}]        [{x[0]: .3f},{x[1]: .3f},{x[2]: .3f}]  z=[{z[0]: .3f},{z[1]: .3f},{z[2]: .3f}]")

if __name__ == '__main__':
    main(sys.argv[1])
