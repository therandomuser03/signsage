#!/usr/bin/env python3
"""Dump a GLB's node hierarchy, skins, and bind poses to JSON (and markdown)."""
import json, struct, sys, math

def read_glb(path):
    with open(path, 'rb') as f:
        magic, version, length = struct.unpack('<III', f.read(12))
        assert magic == 0x46546C67, 'not a GLB'
        chunks = []
        while f.tell() < length:
            clen, ctype = struct.unpack('<II', f.read(8))
            chunks.append((ctype, f.read(clen)))
    js = next(c for t, c in chunks if t == 0x4E4F534A)
    return json.loads(js)

def main(path, out_json):
    g = read_glb(path)
    nodes = g.get('nodes', [])
    skins = g.get('skins', [])
    scene = g.get('scenes', [{}])[g.get('scene', 0)]
    children = {i: n.get('children', []) for i, n in enumerate(nodes)}
    parent = {}
    for i, ch in children.items():
        for c in ch:
            parent[c] = i

    def node_name(i):
        return nodes[i].get('name', f'#{i}')

    # skin joint sets
    skin_joints = set()
    for s in skins:
        skin_joints.update(s.get('joints', []))

    bones = []
    for i, n in enumerate(nodes):
        is_joint = i in skin_joints
        t = n.get('translation', [0,0,0]); r = n.get('rotation', [0,0,0,1]); s = n.get('scale', [1,1,1])
        bones.append({
            'index': i, 'name': node_name(i), 'parent': node_name(parent[i]) if i in parent else None,
            'is_joint': is_joint,
            'translation': t, 'rotation_xyzw': r, 'scale': s,
            'has_mesh': 'mesh' in n,
        })

    # depth-first order from scene roots
    order = []
    def walk(i, depth):
        order.append((i, depth))
        for c in children.get(i, []):
            walk(c, depth + 1)
    for root in scene.get('nodes', []):
        walk(root, 0)

    md = []
    md.append(f'# GLB skeleton dump\n\nSource: `{path}`\n')
    md.append(f'- nodes: {len(nodes)}, skins: {len(skins)}, meshes: {len(g.get("meshes", []))}, animations: {len(g.get("animations", []))}\n')
    if skins:
        md.append(f'- skin joints: {len(skin_joints)}\n')
    md.append('\n## Node hierarchy (depth-first)\n\n```\n')
    for i, depth in order:
        n = nodes[i]
        tag = ' [JOINT]' if i in skin_joints else ''
        md.append('  ' * depth + node_name(i) + tag + '\n')
    md.append('```\n\n## Per-node rest local TRS\n\n')
    for i, depth in order:
        b = bones[i]
        md.append(f"- `{b['name']}` parent=`{b['parent']}` t={fmt(b['translation'])} r(xyzw)={fmt(b['rotation_xyzw'], 6)} s={fmt(b['scale'])}\n")

    md.append('\n## Skins\n\n')
    for si, s in enumerate(skins):
        md.append(f'### skin {si}: {len(s.get("joints", []))} joints, skeleton root node: {node_name(s["skeleton"]) if "skeleton" in s else "n/a"}\n\n')

    with open(out_json, 'w') as f:
        json.dump({'file': path, 'nodes': bones, 'skins': skins, 'animations': [a.get('name') for a in g.get('animations', [])]}, f, indent=2)
    return ''.join(md)

def fmt(v, nd=4):
    return '[' + ', '.join(f'{x:.{nd}f}' for x in v) + ']'

if __name__ == '__main__':
    md = main(sys.argv[1], sys.argv[2])
    with open(sys.argv[3], 'w') as f:
        f.write(md)
