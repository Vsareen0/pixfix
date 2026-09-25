"""Exemplar-based inpainting with PatchMatch (classical, no training needed).

Idea: instead of blurring colours inward (Telea / Navier–Stokes), fill the hole
by *copying real patches* from the known part of the image, chosen so that
overlapping patches agree with each other.

Algorithm – "Space-Time Completion" (Wexler, Shechtman & Irani, 2007) with the
PatchMatch nearest-neighbour search (Barnes et al., SIGGRAPH 2009), the same
family of methods behind Photoshop's original Content-Aware Fill:

  1. Build an image pyramid and start at a scale where the hole is only a few
     pixels wide (so coarse structure is easy to get right).
  2. At each scale repeat (EM iterations):
       a. NNF step – for every patch overlapping the hole, find the most
          similar fully-known patch (PatchMatch: random init → propagate good
          matches to neighbours → random search at shrinking radii).
       b. Vote step – every hole pixel becomes a weighted average of the
          pixels that its covering patches' matches suggest.
  3. Up-sample the nearest-neighbour field to the next scale and repeat.

The implementation is vectorised NumPy: propagation is done for all patches in
parallel (a "jump-flood" variant of PatchMatch's sequential scan), which is
what makes it fast enough in pure Python.
"""
import cv2
import numpy as np
from scipy import ndimage

PATCH_RADIUS = 3  # 7x7 patches


class PatchMatchInpainter:
    name = "patchmatch"

    def __init__(self, patch_radius: int = PATCH_RADIUS, max_side: int = 1024, seed: int = 0,
                 priority: bool = True, min_coarse: int = 64, poisson: bool = False):
        self.r = patch_radius
        self.poisson = poisson
        self.min_coarse = min_coarse  # smallest side (px) allowed at the coarsest pyramid level
        self.priority = priority
        self.max_side = max_side
        self.seed = seed

    # ------------------------------------------------------------------ API
    def __call__(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """img: HxWx3 uint8 RGB, mask: HxW uint8 (255 = fill). Returns HxWx3 uint8."""
        h0, w0 = img.shape[:2]
        hole = mask > 127
        if not hole.any():
            return img.copy()

        # Work at a capped resolution; the engine blends the result back into
        # the full-resolution original, so only the filled region is affected.
        scale = min(1.0, self.max_side / max(h0, w0))
        if scale < 1:
            size = (max(1, round(w0 * scale)), max(1, round(h0 * scale)))
            work = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            hole_w = cv2.resize(hole.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
        else:
            work, hole_w = img, hole

        lab = cv2.cvtColor(work, cv2.COLOR_RGB2LAB).astype(np.float32)
        out_lab = self._complete(lab, hole_w)
        out = cv2.cvtColor(np.clip(out_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)

        if scale < 1:
            out = cv2.resize(out, (w0, h0), interpolation=cv2.INTER_CUBIC)
        result = img.copy()
        result[hole] = out[hole]
        if self.poisson:
            result = poisson_blend(result, img, hole)
        return result

    # ------------------------------------------------------------- pyramid
    def _complete(self, lab: np.ndarray, hole: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        r = self.r

        # Number of levels: coarsest hole "thickness" ≈ patch radius,
        # but keep the coarsest image comfortably larger than a few patches.
        thickness = ndimage.distance_transform_edt(hole).max()
        levels = 1
        while (thickness / 2 ** levels > r and min(hole.shape) / 2 ** levels >= self.min_coarse):
            levels += 1

        imgs, holes = [lab], [hole]
        for _ in range(levels - 1):
            im, ho = self._downsample(imgs[-1], holes[-1])
            imgs.append(im)
            holes.append(ho)

        nnf = None
        for lvl in range(levels - 1, -1, -1):
            im, ho = imgs[lvl].copy(), holes[lvl]
            level = _Level(im, ho, r, self.priority)
            if level.n_targets == 0 or level.n_sources == 0:
                continue
            if nnf is None:
                # Coarsest level: "onion-peel" initial fill from the hole boundary
                # inward, matching on known pixels only, so the object being
                # removed never influences the first guess.
                level.onion_peel(rng)
                level.random_nnf(rng)
            else:
                level.init_from_coarse(nnf)
                level.vote(uniform=True)  # initial fill from the up-sampled matches

            coarse = lvl == levels - 1
            # Coarse levels decide the structure -> search widely and iterate a lot.
            # Finer levels start from the up-sampled matches, which are already
            # close, so a local search is enough (this keeps large holes fast).
            em_iters = 8 if coarse else (2 if lvl == 0 else 4)
            max_radius = None if coarse or lvl >= levels - 2 else 16
            for _ in range(em_iters):
                level.patchmatch(rng, iters=4 if coarse else 2, max_radius=max_radius)
                level.vote()
            nnf = level.full_nnf()
            imgs[lvl] = level.image()
        return imgs[0]

    @staticmethod
    def _downsample(im, hole):
        """Half-size image where hole pixels do not bleed into known ones."""
        h, w = hole.shape
        size = (max(1, (w + 1) // 2), max(1, (h + 1) // 2))
        known = (~hole).astype(np.float32)
        num = cv2.resize(im * known[..., None], size, interpolation=cv2.INTER_AREA)
        den = cv2.resize(known, size, interpolation=cv2.INTER_AREA)
        small = num / np.maximum(den, 1e-6)[..., None]
        small_hole = den < 0.999  # any hole pixel in the 2x2 block -> hole
        return small.astype(np.float32), small_hole


def poisson_blend(filled: np.ndarray, original: np.ndarray, hole: np.ndarray, pad: int = 16):
    """Optional gradient-domain clean-up (Pérez et al., "Poisson Image Editing", 2003).

    Keeps the *gradients* of the fill but solves for colours that match the hole
    boundary, which removes brightness offsets between copied patches and their
    surroundings. Off by default: in our tests it also dragged leftover object
    pixels on the mask edge (glows, shadows) into the fill, which was usually worse.
    Enable with PatchMatchInpainter(poisson=True) to compare.
    """
    m = cv2.dilate(hole.astype(np.uint8) * 255, np.ones((5, 5), np.uint8))
    # seamlessClone needs the mask away from the image border -> work on a padded copy
    src = cv2.copyMakeBorder(filled, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    dst = cv2.copyMakeBorder(original, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    mp = cv2.copyMakeBorder(m, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    x, y, w, h = cv2.boundingRect(mp)
    try:
        out = cv2.seamlessClone(src, dst, mp, (x + w // 2, y + h // 2), cv2.NORMAL_CLONE)
    except cv2.error:
        return filled
    out = out[pad:-pad, pad:-pad]
    result = original.copy()
    sel = m > 0
    result[sel] = out[sel]
    return result


class _Level:
    """State for one pyramid level. Works on a reflect-padded copy of the image."""

    def __init__(self, im, hole, r, priority=True):
        self.r = r
        self.priority = priority
        self.h, self.w = hole.shape
        self.P = np.pad(im, ((r, r), (r, r), (0, 0)), mode="reflect")
        ph = np.pad(hole, r, mode="reflect")
        self.hole_p = ph
        hp, wp = ph.shape

        # Valid source centres: whole patch known and inside the padded image.
        k = 2 * r + 1
        known_patch = cv2.erode((~ph).astype(np.uint8), np.ones((k, k), np.uint8),
                                borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
        inner = np.zeros_like(known_patch)
        inner[r:hp - r, r:wp - r] = True
        valid = known_patch & inner
        self.n_sources = int(valid.sum())
        self.snap_valid = valid
        # For every pixel, index of the nearest valid source centre ("snap").
        _, (self.snap_y, self.snap_x) = ndimage.distance_transform_edt(~valid, return_indices=True)

        # Targets: patch centres (inside the unpadded image) whose patch touches the hole.
        touches = cv2.dilate(ph.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
        touches[:r, :] = touches[hp - r:, :] = False
        touches[:, :r] = touches[:, wp - r:] = False
        self.ty, self.tx = np.nonzero(touches)
        self.n_targets = len(self.ty)
        self.idx = -np.ones((hp, wp), np.int64)
        self.idx[self.ty, self.tx] = np.arange(self.n_targets)

        dy, dx = np.mgrid[-r:r + 1, -r:r + 1]
        self.dy, self.dx = dy.ravel(), dx.ravel()
        self.fill = ph  # pixels that get rewritten by voting
        self.ny = self.nx = self.dist = None

    # ---------------------------------------------------------------- utils
    def set_image(self, im):
        r = self.r
        self.P = np.pad(im, ((r, r), (r, r), (0, 0)), mode="reflect")

    def image(self):
        r = self.r
        return self.P[r:r + self.h, r:r + self.w]

    def _snap(self, y, x):
        hp, wp = self.hole_p.shape
        y = np.clip(y, 0, hp - 1)
        x = np.clip(x, 0, wp - 1)
        return self.snap_y[y, x], self.snap_x[y, x]

    def _target_patches(self):
        return self._gather(self.ty, self.tx)  # N,K,3

    def _gather(self, cy, cx):
        """All patches centred at (cy, cx) -> N,K,3 (flat-index take is ~2x faster than 2-D fancy indexing)."""
        wp = self.P.shape[1]
        flat = (cy * wp + cx)[:, None] + (self.dy * wp + self.dx)[None, :]
        return np.take(self.P.reshape(-1, 3), flat, axis=0)

    def _dist(self, A, cy, cx, W=None):
        B = self._gather(cy, cx)
        D = A - B
        if W is None:
            return np.einsum("nkc,nkc->n", D, D)
        # SSD over known pixels only, rescaled to a full-patch equivalent
        return np.einsum("nkc,nkc,nk->n", D, D, W) * (W.shape[1] / np.maximum(W.sum(1), 1))

    # ------------------------------------------------------- initial fill
    def onion_peel(self, rng, max_sources=6000):
        """Fill the hole ring by ring (outside -> in). Each ring pixel takes the
        centre colour of the best-matching source patch, compared on the pixels
        already known. Exact (brute-force) search: only used at the tiny
        coarsest level, written as a single matrix product per ring."""
        sy, sx = np.nonzero(self.snap_valid)
        if len(sy) > max_sources:
            pick = rng.choice(len(sy), max_sources, replace=False)
            sy, sx = sy[pick], sx[pick]
        S = self.P[sy[:, None] + self.dy, sx[:, None] + self.dx].reshape(len(sy), -1)  # M, K*3
        S2 = (S.reshape(len(sy), -1, 3) ** 2).sum(-1)  # M, K

        known = ~self.hole_p.copy()
        hp, wp = known.shape
        r = self.r
        cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
        while True:
            ring = (cv2.dilate(known.astype(np.uint8), cross) > 0) & ~known
            ring[:r, :] = ring[hp - r:, :] = False
            ring[:, :r] = ring[:, wp - r:] = False
            fy, fx = np.nonzero(ring)
            if len(fy) == 0:
                break
            W = known[fy[:, None] + self.dy, fx[:, None] + self.dx].astype(np.float32)  # F, K
            # Confidence priority (Criminisi et al. 2004): fill the best-supported
            # ring pixels first, so strong structures are extended before weak ones.
            if self.priority:
                conf = W.sum(1)
                keep = conf >= np.median(conf)
                fy, fx, W = fy[keep], fx[keep], W[keep]
            T = self.P[fy[:, None] + self.dy, fx[:, None] + self.dx]            # F, K, 3
            WT = (W[..., None] * T).reshape(len(fy), -1)                         # F, K*3
            # sum_k W (T - S)^2 = sum W T^2 - 2 (W T)·S + W·S^2   (T^2 term is constant per row)
            d = -2 * WT @ S.T + W @ S2.T                                         # F, M
            best = d.argmin(1)
            self.P[fy, fx] = self.P[sy[best], sx[best]]
            known[fy, fx] = True
        # pixels in the pad border that were never reached keep their values

    # ---------------------------------------------------------------- NNF
    def random_nnf(self, rng):
        hp, wp = self.hole_p.shape
        self.ny, self.nx = self._snap(rng.integers(0, hp, self.n_targets),
                                      rng.integers(0, wp, self.n_targets))
        self.dist = self._dist(self._target_patches(), self.ny, self.nx)

    def init_from_coarse(self, coarse_nnf):
        """coarse_nnf: (cy, cx) full-size arrays (unpadded coords) of the coarser level."""
        cy_full, cx_full = coarse_nnf
        r = self.r
        uy, ux = self.ty - r, self.tx - r  # unpadded target coords
        ch, cw = cy_full.shape
        py = np.clip(uy // 2, 0, ch - 1)
        px = np.clip(ux // 2, 0, cw - 1)
        y = cy_full[py, px] * 2 + (uy % 2) + r
        x = cx_full[py, px] * 2 + (ux % 2) + r
        self.ny, self.nx = self._snap(y, x)

    def full_nnf(self):
        """NNF in unpadded coordinates for every pixel (identity outside targets)."""
        r = self.r
        yy, xx = np.mgrid[0:self.h, 0:self.w]
        cy, cx = yy.copy(), xx.copy()
        cy[self.ty - r, self.tx - r] = self.ny - r
        cx[self.ty - r, self.tx - r] = self.nx - r
        return cy, cx

    def _try(self, A, cy, cx, W=None):
        cy, cx = self._snap(cy, cx)
        d = self._dist(A, cy, cx, W)
        better = d < self.dist
        self.ny[better], self.nx[better], self.dist[better] = cy[better], cx[better], d[better]

    def patchmatch(self, rng, iters=2, known_only=False, max_radius=None):
        A = self._target_patches()
        W = None
        if known_only:
            W = (~self.hole_p[self.ty[:, None] + self.dy, self.tx[:, None] + self.dx]).astype(np.float32)
        self.dist = self._dist(A, self.ny, self.nx, W)
        hp, wp = self.hole_p.shape
        for it in range(iters):
            # Propagation: adopt a neighbour's match, shifted by the neighbour offset.
            steps = (4, 2, 1) if it == 0 and max_radius is None else (1,)
            for s in steps:
                for oy, ox in ((0, -s), (0, s), (-s, 0), (s, 0)):
                    nb = self.idx[np.clip(self.ty + oy, 0, hp - 1), np.clip(self.tx + ox, 0, wp - 1)]
                    ok = nb >= 0
                    if not ok.any():
                        continue
                    cy = np.where(ok, self.ny[np.maximum(nb, 0)] - oy, self.ny)
                    cx = np.where(ok, self.nx[np.maximum(nb, 0)] - ox, self.nx)
                    self._try(A, cy, cx, W)
            # Random search around the current best at exponentially shrinking radii.
            radius = max(hp, wp) if max_radius is None else min(max_radius, max(hp, wp))
            while radius >= 1:
                cy = self.ny + rng.integers(-radius, radius + 1, self.n_targets)
                cx = self.nx + rng.integers(-radius, radius + 1, self.n_targets)
                self._try(A, cy, cx, W)
                radius //= 2
        if known_only:
            self.dist = None  # distances are not comparable with full-patch ones

    # ---------------------------------------------------------------- vote
    def vote(self, uniform=False):
        """Weighted average of overlapping patch suggestions, only on hole pixels.

        uniform=True is used right after up-sampling, when the hole still holds
        the original object's pixels and patch distances are not meaningful yet.
        """
        if uniform:
            w = np.ones(self.n_targets, np.float32)
        else:
            if self.dist is None:
                self.dist = self._dist(self._target_patches(), self.ny, self.nx)
            # Similar patches get more say (robust sigma from the distance distribution).
            sigma2 = max(float(np.percentile(self.dist, 75)), 1e-3)
            w = np.exp(-self.dist / (2 * sigma2)).astype(np.float32) + 1e-6

        acc = np.zeros_like(self.P)
        wsum = np.zeros(self.P.shape[:2], np.float32)
        for dy, dx in zip(self.dy, self.dx):
            y, x = self.ty + dy, self.tx + dx
            # (y, x) pairs are unique for a fixed offset -> plain fancy-index add is safe
            acc[y, x] += w[:, None] * self.P[self.ny + dy, self.nx + dx]
            wsum[y, x] += w
        upd = self.fill & (wsum > 0)
        self.P[upd] = acc[upd] / wsum[upd][:, None]
        self.dist = None
