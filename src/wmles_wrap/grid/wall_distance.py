"""Distance from cell centroids to the nearest wall, on the device with Warp's BVH.

1. The wall surface is every parent face with a ``wall`` or ``wall_modeled`` BC, each quad
   split into two triangles (zero-area triangles of collapsed faces are dropped). Each device
   holds the whole surface in one ``wp.Mesh``.
2. ``k_select`` (one thread per cell, padded range): the float32 BVH (``wp.Mesh`` points are
   vec3f) finds a closest triangle for the float32-rounded centroid. Rounding can make it pick
   a neighbour of the true closest triangle (near-ties at shared edges, also across blocks),
   so a float32 box query of half-size d + 4 delta around the point (delta bounds the float32
   rounding of the coordinates) re-checks the candidates in float64. The candidate loop is
   capped (``max_candidates``); a partial check can only lower the distance.
3. ``k_distance``: the float64 point-to-triangle distance to the selected triangle, from float64
   centroids and vertices, stored in the run precision. With the triangle frozen, d is a smooth
   function of the node coordinates (adjoint design rule 3).

Coordinates are shifted by the centre of the wall surface before the float32 cast, which keeps
float32 magnitudes, and so delta, as small as possible.
"""

# NOTE: no "from __future__ import annotations": Warp must resolve closure types in kernel signatures.

from dataclasses import dataclass

import numpy as np
import warp as wp

from ..precision import Precision, kernel_factory

WALL_BC_TYPES = ("wall", "wall_modeled")
NO_WALL_DISTANCE = 1.0e30  # stored where there is no wall surface at all
EPS32 = float(np.finfo(np.float32).eps)


@dataclass
class WallSurface:
    verts: np.ndarray  # (nv, 3) float64
    tris: np.ndarray  # (nt, 3) int32
    origin: np.ndarray  # (3,) float64 centre of the bounding box

    @property
    def radius(self) -> float:
        """Max |vertex - origin| (inf-norm), the float32 magnitude scale of the mesh."""
        return float(np.abs(self.verts - self.origin).max()) if len(self.verts) else 0.0


def wall_faces(subs) -> list[tuple[int, int]]:
    """(parent block, face) of all faces with a wall BC, sorted."""
    return sorted({(s.parent, f) for s in subs for f, bc in s.face_bc.items()
                   if bc is not None and bc.type in WALL_BC_TYPES})


def wall_surface(parent_blocks: list[np.ndarray], faces: list[tuple[int, int]]) -> WallSurface:
    """Triangulate the wall faces of the parent blocks: quad (a, b, c, d) -> (a, b, c), (a, c, d)."""
    verts, tris, nv = [], [], 0
    for b, f in faces:
        x = parent_blocks[b]
        F = np.take(x, 0 if f % 2 == 0 else x.shape[f // 2] - 1, axis=f // 2)  # (n1, n2, 3)
        n1, n2 = F.shape[:2]
        idx = np.arange(n1 * n2).reshape(n1, n2) + nv
        a, bb, c, d = idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]
        t = np.concatenate([np.stack([a, bb, c], -1).reshape(-1, 3), np.stack([a, c, d], -1).reshape(-1, 3)])
        verts.append(F.reshape(-1, 3))
        tris.append(t)
        nv += n1 * n2
    if not verts:
        return WallSurface(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32), np.zeros(3))
    V = np.concatenate(verts).astype(np.float64)
    T = np.concatenate(tris)
    e1, e2 = V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]]
    area2 = np.linalg.norm(np.cross(e1, e2), axis=1)
    keep = area2 > 1e-12 * (np.sum(e1 * e1, 1) + np.sum(e2 * e2, 1))
    T = T[keep].astype(np.int32)
    origin = 0.5 * (V.min(0) + V.max(0))
    return WallSurface(V, T, origin)


@kernel_factory
def _wd_kernels(prec: Precision):
    scalar = prec.scalar
    f64 = wp.float64
    zero, one, eighth = f64(0.0), f64(1.0), f64(0.125)

    @wp.func
    def tri_dist2(p: wp.vec3d, a: wp.vec3d, b: wp.vec3d, c: wp.vec3d):
        """Squared distance from p to triangle abc (Ericson, Real-Time Collision Detection 5.1.5)."""
        ab = b - a
        ac = c - a
        ap = p - a
        d1 = wp.dot(ab, ap)
        d2 = wp.dot(ac, ap)
        bp = p - b
        d3 = wp.dot(ab, bp)
        d4 = wp.dot(ac, bp)
        cp = p - c
        d5 = wp.dot(ab, cp)
        d6 = wp.dot(ac, cp)
        vc = d1 * d4 - d3 * d2
        vb = d5 * d2 - d1 * d6
        va = d3 * d6 - d5 * d4
        q = a
        if d1 <= zero and d2 <= zero:
            q = a
        elif d3 >= zero and d4 <= d3:
            q = b
        elif vc <= zero and d1 >= zero and d3 <= zero:
            q = a + (d1 / (d1 - d3)) * ab
        elif d6 >= zero and d5 <= d6:
            q = c
        elif vb <= zero and d2 >= zero and d6 <= zero:
            q = a + (d2 / (d2 - d6)) * ac
        elif va <= zero and d4 - d3 >= zero and d5 - d6 >= zero:
            q = b + ((d4 - d3) / ((d4 - d3) + (d5 - d6))) * (c - b)
        else:
            den = one / (va + vb + vc)
            q = a + (vb * den) * ab + (vc * den) * ac
        r = p - q
        return wp.dot(r, r)

    @wp.kernel(module="unique")
    def k_centroids(X: wp.array3d(dtype=wp.vec3d), xc: wp.array3d(dtype=wp.vec3d)):
        p, q, r = wp.tid()
        xc[p, q, r] = eighth * (
            X[p, q, r] + X[p + 1, q, r] + X[p, q + 1, r] + X[p + 1, q + 1, r]
            + X[p, q, r + 1] + X[p + 1, q, r + 1] + X[p, q + 1, r + 1] + X[p + 1, q + 1, r + 1]
        )

    @wp.kernel(module="unique")
    def k_select(
        mesh: wp.uint64,
        xc: wp.array3d(dtype=wp.vec3d),
        verts: wp.array(dtype=wp.vec3d),
        tris: wp.array(dtype=wp.vec3i),
        origin: wp.vec3d,
        radius: float,
        eps32: float,
        max_candidates: int,
        face: wp.array3d(dtype=wp.int32),
    ):
        p, q, r = wp.tid()
        x = xc[p, q, r]
        xs = x - origin
        p32 = wp.vec3f(float(xs[0]), float(xs[1]), float(xs[2]))
        best = int(-1)
        best_d2 = f64(0.0)
        hit = wp.mesh_query_point_no_sign(mesh, p32, 3.0e38)
        if hit.result:
            best = hit.face
            t = tris[best]
            best_d2 = tri_dist2(x, verts[t[0]], verts[t[1]], verts[t[2]])
            # float32 rounding bound of point and mesh coordinates
            mag = wp.max(wp.max(wp.abs(p32[0]), wp.abs(p32[1])), wp.max(wp.abs(p32[2]), radius))
            h = float(wp.sqrt(best_d2)) * (1.0 + 4.0 * eps32) + 4.0 * eps32 * mag
            hv = wp.vec3f(h, h, h)
            query = wp.mesh_query_aabb(mesh, p32 - hv, p32 + hv)
            cand = int(0)
            n = int(0)
            while wp.mesh_query_aabb_next(query, cand):
                if n < max_candidates:
                    tc = tris[cand]
                    d2 = tri_dist2(x, verts[tc[0]], verts[tc[1]], verts[tc[2]])
                    if d2 < best_d2 or (d2 == best_d2 and cand < best):
                        best_d2 = d2
                        best = cand
                n += 1
        face[p, q, r] = wp.int32(best)

    @wp.kernel(module="unique")
    def k_distance(
        xc: wp.array3d(dtype=wp.vec3d),
        face: wp.array3d(dtype=wp.int32),
        verts: wp.array(dtype=wp.vec3d),
        tris: wp.array(dtype=wp.vec3i),
        no_wall: f64,
        d: wp.array3d(dtype=scalar),
    ):
        p, q, r = wp.tid()
        f = int(face[p, q, r])
        dist = no_wall
        if f >= 0:
            t = tris[f]
            dist = wp.sqrt(tri_dist2(xc[p, q, r], verts[t[0]], verts[t[1]], verts[t[2]]))
        d[p, q, r] = scalar(dist)

    return k_centroids, k_select, k_distance


@dataclass
class WallDistance:
    """Per sub-block (padded cells): nearest wall triangle (-1: no wall) and distance."""

    face: dict[int, wp.array]
    d: dict[int, wp.array]
    surface: WallSurface


def compute_wall_distance(subs, parent_blocks, prec: Precision, ng: int, max_candidates: int = 256) -> WallDistance:
    """Wall distance at the padded cell centroids of every sub-block (``sub.xp`` must be set)."""
    k_centroids, k_select, k_distance = _wd_kernels(prec)
    surf = wall_surface(parent_blocks, wall_faces(subs))
    meshes = {}
    if len(surf.tris):
        for dev in sorted({s.device for s in subs}):
            pts = wp.array((surf.verts - surf.origin).astype(np.float32), dtype=wp.vec3f, device=dev)
            idx = wp.array(surf.tris.ravel(), dtype=wp.int32, device=dev)
            v64 = wp.array(surf.verts, dtype=wp.vec3d, device=dev)
            t3 = wp.array(surf.tris, dtype=wp.vec3i, device=dev)
            meshes[dev] = (wp.Mesh(points=pts, indices=idx), pts, idx, v64, t3)
    faces, dists = {}, {}
    for s in subs:
        dev = s.device
        nc = s.padded_shape_cells(ng)
        X = wp.array(np.asarray(s.xp, dtype=np.float64), dtype=wp.vec3d, device=dev)
        xc = wp.zeros(nc, dtype=wp.vec3d, device=dev)
        wp.launch(k_centroids, dim=nc, inputs=[X], outputs=[xc], device=dev)
        face = wp.full(nc, -1, dtype=wp.int32, device=dev)
        d = wp.zeros(nc, dtype=prec.scalar, device=dev)
        if dev in meshes:
            mesh, _, _, v64, t3 = meshes[dev]
            wp.launch(k_select, dim=nc,
                      inputs=[mesh.id, xc, v64, t3, wp.vec3d(*surf.origin), float(surf.radius), EPS32,
                              int(max_candidates)],
                      outputs=[face], device=dev)
            wp.launch(k_distance, dim=nc, inputs=[xc, face, v64, t3, wp.float64(NO_WALL_DISTANCE)],
                      outputs=[d], device=dev)
        else:
            d.fill_(NO_WALL_DISTANCE)
        faces[s.id], dists[s.id] = face, d
    return WallDistance(faces, dists, surf)
