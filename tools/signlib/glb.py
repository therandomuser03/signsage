"""Minimal GLB rest-pose reader + forward kinematics, no dependencies beyond numpy."""
import json, struct
import numpy as np

def read_glb_json(path):
    with open(path, 'rb') as f:
        magic, version, length = struct.unpack('<III', f.read(12))
        assert magic == 0x46546C67, 'not a GLB'
        chunks = []
        while f.tell() < length:
            clen, ctype = struct.unpack('<II', f.read(8))
            chunks.append((ctype, f.read(clen)))
    return json.loads(next(c for t, c in chunks if t == 0x4E4F534A))

def quat_to_mat(q):
    x, y, z, w = q
    n = x*x + y*y + z*z + w*w
    if n == 0:
        return np.eye(3)
    s = 2.0 / n
    xx, yy, zz = x*x*s, y*y*s, z*z*s
    xy, xz, yz = x*y*s, x*z*s, y*z*s
    wx, wy, wz = w*x*s, w*y*s, w*z*s
    return np.array([
        [1-(yy+zz), xy-wz, xz+wy],
        [xy+wz, 1-(xx+zz), yz-wx],
        [xz-wy, yz+wx, 1-(xx+yy)],
    ])

def mat_to_quat(m):
    t = m[0,0] + m[1,1] + m[2,2]
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        return np.array([(m[2,1]-m[1,2])/s, (m[0,2]-m[2,0])/s, (m[1,0]-m[0,1])/s, s/4])
    i = int(np.argmax([m[0,0], m[1,1], m[2,2]]))
    if i == 0:
        s = np.sqrt(1.0 + m[0,0] - m[1,1] - m[2,2]) * 2
        return np.array([s/4, (m[0,1]+m[1,0])/s, (m[0,2]+m[2,0])/s, (m[2,1]-m[1,2])/s])
    if i == 1:
        s = np.sqrt(1.0 + m[1,1] - m[0,0] - m[2,2]) * 2
        return np.array([(m[0,1]+m[1,0])/s, s/4, (m[1,2]+m[2,1])/s, (m[0,2]-m[2,0])/s])
    s = np.sqrt(1.0 + m[2,2] - m[0,0] - m[1,1]) * 2
    return np.array([(m[0,2]+m[2,0])/s, (m[1,2]+m[2,1])/s, s/4, (m[1,0]-m[0,1])/s])

def quat_mul(a, b):
    """Hamilton product a*b (apply b, then a), xyzw order."""
    ax, ay, az, aw = a; bx, by, bz, bw = b
    return np.array([
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
        aw*bw - ax*bx - ay*by - az*bz,
    ])

def quat_conj(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])

def quat_normalize(q):
    n = np.linalg.norm(q)
    return q / n if n > 0 else np.array([0, 0, 0, 1.0])

def quat_slerp(a, b, t):
    a = quat_normalize(a); b = quat_normalize(b)
    d = float(np.dot(a, b))
    if d < 0:
        b = -b; d = -d
    if d > 0.9995:
        return quat_normalize(a + t * (b - a))
    th = np.arccos(np.clip(d, -1, 1))
    s = np.sin(th)
    return (a * np.sin((1-t)*th) + b * np.sin(t*th)) / s

def quat_from_axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    n = np.linalg.norm(axis)
    if n == 0:
        return np.array([0, 0, 0, 1.0])
    axis = axis / n
    h = angle / 2.0
    return np.array([axis[0]*np.sin(h), axis[1]*np.sin(h), axis[2]*np.sin(h), np.cos(h)])

def swing_arc(from_dir, to_dir):
    """Minimal rotation mapping from_dir onto to_dir (both unit)."""
    f = np.asarray(from_dir, dtype=np.float64); t = np.asarray(to_dir, dtype=np.float64)
    f = f / (np.linalg.norm(f) + 1e-12); t = t / (np.linalg.norm(t) + 1e-12)
    d = float(np.dot(f, t))
    if d > 0.999999:
        return np.array([0, 0, 0, 1.0])
    if d < -0.999999:
        axis = np.cross(f, np.array([1.0, 0, 0]))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(f, np.array([0, 1.0, 0]))
        axis /= np.linalg.norm(axis)
        return quat_from_axis_angle(axis, np.pi)
    axis = np.cross(f, t)
    return quat_from_axis_angle(axis, np.arccos(np.clip(d, -1, 1)))

def quat_rotate(q, v):
    return quat_to_mat(q) @ np.asarray(v, dtype=np.float64)

class Rig:
    """Rest pose of the GLB with forward kinematics."""

    def __init__(self, glb_path):
        g = read_glb_json(glb_path)
        self.nodes = g['nodes']
        self.names = [n.get('name', f'#{i}') for i, n in enumerate(self.nodes)]
        self.by_name = {n: i for i, n in enumerate(self.names)}
        self.parent = {}
        for i, n in enumerate(self.nodes):
            for c in n.get('children', []):
                self.parent[c] = i
        self.rest_local_q = [np.array(n.get('rotation', [0, 0, 0, 1]), dtype=np.float64) for n in self.nodes]
        self.rest_local_t = [np.array(n.get('translation', [0, 0, 0]), dtype=np.float64) for n in self.nodes]
        self.children = {i: self.nodes[i].get('children', []) for i in range(len(self.nodes))}
        # world rest transforms
        self.rest_world_q = [None] * len(self.nodes)
        self.rest_world_p = [None] * len(self.nodes)
        for i in range(len(self.nodes)):
            self._compute_world(i)

    def _compute_world(self, i):
        if self.rest_world_q[i] is not None:
            return
        lt = self.rest_local_t[i]
        lq = self.rest_local_q[i]
        if i in self.parent:
            p = self.parent[i]
            self._compute_world(p)
            self.rest_world_p[i] = self.rest_world_p[p] + quat_rotate(self.rest_world_q[p], lt)
            self.rest_world_q[i] = quat_normalize(quat_mul(self.rest_world_q[p], lq))
        else:
            self.rest_world_p[i] = lt.copy()
            self.rest_world_q[i] = quat_normalize(lq)

    def rest_world_dir(self, name):
        """World-space direction of the bone's local +Y (length axis) at rest."""
        return quat_rotate(self.rest_world_q[self.by_name[name]], np.array([0, 1, 0]))

    def world_positions(self, local_quats):
        """FK with per-node local quaternion overrides (dict name->xyzw)."""
        n = len(self.nodes)
        out_q = [None] * n
        out_p = [None] * n
        for i in range(n):
            name = self.names[i]
            lq = np.array(local_quats.get(name, self.rest_local_q[i]), dtype=np.float64)
            lt = self.rest_local_t[i]
            if i in self.parent:
                p = self.parent[i]
                # parent already computed (nodes are in depth order in this GLB; enforce anyway)
                if out_q[p] is None:
                    self._fk_node(p, local_quats, out_q, out_p)
                out_p[i] = out_p[p] + quat_rotate(out_q[p], lt)
                out_q[i] = quat_normalize(quat_mul(out_q[p], lq))
            else:
                out_p[i] = lt.copy()
                out_q[i] = quat_normalize(lq)
        return {self.names[i]: out_p[i] for i in range(n)}, {self.names[i]: out_q[i] for i in range(n)}

    def _fk_node(self, i, local_quats, out_q, out_p):
        if out_q[i] is not None:
            return
        name = self.names[i]
        lq = np.array(local_quats.get(name, self.rest_local_q[i]), dtype=np.float64)
        lt = self.rest_local_t[i]
        if i in self.parent:
            p = self.parent[i]
            if out_q[p] is None:
                self._fk_node(p, local_quats, out_q, out_p)
            out_p[i] = out_p[p] + quat_rotate(out_q[p], lt)
            out_q[i] = quat_normalize(quat_mul(out_q[p], lq))
        else:
            out_p[i] = lt.copy()
            out_q[i] = quat_normalize(lq)
