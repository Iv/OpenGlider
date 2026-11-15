from openglider.airfoil import Profile3D
from openglider.vector.spline import Bezier


class MiniRib:
    def __init__(self, yvalue, front_cut, back_cut=1, func=None, name="minirib", te_factor=None, turn=None,
                 front_cut_lower=None, blend=0.03):
        """
        turn (deg): the designer's trailing edge minirib (Tayran-8, tayran8_v4.m): the section is the cell's
        unballooned section at yvalue; behind front_cut its upper edge is turned up by `turn` about the upper
        trailing edge point (in the section plane), the lower edge (behind front_cut_lower) stays unballooned;
        over `blend` (chord fraction) in front of the cuts the panel goes from the ballooned cell to the part.
        func(x): share of the ballooning kept at the minirib (0..1, x = chord fraction); the cell is split into
        two child cells there, each ballooned so that the panel keeps its length (bulges left and right of it).
        te_factor: a trailing edge minirib (back_cut 1): the ballooning goes by a monotone spline from 1 at the
        front_cut (flat there) to te_factor at the first third of the minirib and to 0 at the trailing edge
        (e.g. 0.25: the centre of the cell held at 1/4 of the original bulge, the trailing edge sharp).
        Without it: the Bezier 1 -> 0 within the first third.
        """
        # Profile3D.__init__(self, [], name)
        self.te_factor = te_factor
        self.turn = turn
        self.front_cut_lower = front_cut_lower
        self.blend = blend
        if not func and te_factor is not None and back_cut >= 1 and front_cut > 0:
            from scipy.interpolate import PchipInterpolator

            length = 1.0 - front_cut
            spline = PchipInterpolator(
                [front_cut - 0.1 * length, front_cut, front_cut + length / 3, 1.0],
                [1.0, 1.0, te_factor, 0.0],
            )
            func = lambda x: float(spline(x))  # noqa: E731

        if not func:  # Function is a bezier-function depending on front/back
            if front_cut > 0:
                points = [
                    [front_cut, 1],
                    [front_cut * 2.0 / 3 + back_cut * 1.0 / 3, 0],
                ]  #
            else:
                points = [[front_cut, 0]]

            if back_cut < 1:
                points = points + [
                    [front_cut * 1.0 / 3 + back_cut * 2.0 / 3, 0],
                    [back_cut, 1],
                ]
            else:
                points = points + [[back_cut, 0]]

            func = Bezier(points).interpolation()

        self.__function__ = func

        self.y_value = yvalue
        self.front_cut = front_cut
        self.back_cut = back_cut

    def __json__(self):
        # func is not stored (rebuilt from front / back cut and te_factor)
        d = {"yvalue": self.y_value, "front_cut": self.front_cut, "back_cut": self.back_cut,
             "name": getattr(self, "name", "minirib"), "te_factor": self.te_factor}
        if getattr(self, "turn", None) is not None:
            d.update(turn=self.turn, front_cut_lower=self.front_cut_lower, blend=self.blend)
        return d

    def section(self, x_values, with_bal, without_bal):
        """section points of the part, per profile station (x signed: < 0 upper)"""
        import numpy as np

        W = np.asarray(with_bal, dtype=float)
        F = np.asarray(without_bal, dtype=float)
        xs = np.asarray(x_values, dtype=float)
        if getattr(self, "turn", None) is None:
            fakt = np.array([self.function(x) for x in xs])
            return F + fakt[:, None] * (W - F)
        out = W.copy()
        # section plane: chord (nose -> TE) and the normal of the plane through the section points
        c = F.mean(axis=0)
        _, _, vt = np.linalg.svd(F - c)
        n = vt[2]
        up_idx = np.where(xs < 0)[0]
        te = F[up_idx[np.argmax(np.abs(xs[up_idx]))]]
        a = np.radians(self.turn)

        def rot(p, ang):  # Rodrigues about n through te
            v = p - te
            return te + v * np.cos(ang) + np.cross(n, v) * np.sin(ang) + n * np.dot(n, v) * (1 - np.cos(ang))
        # sign: the turned upper edge must move away from the lower side (up)
        lo_idx = np.where(xs >= 0)[0]
        i_front = up_idx[np.argmin(np.abs(np.abs(xs[up_idx]) - self.front_cut))]
        j_front = lo_idx[np.argmin(np.abs(xs[lo_idx] - self.front_cut))]
        d0 = np.linalg.norm(F[i_front] - F[j_front])
        sgn = 1.0 if np.linalg.norm(rot(F[i_front], a) - F[j_front]) > d0 else -1.0
        fl = self.front_cut_lower if self.front_cut_lower is not None else self.front_cut
        b = max(self.blend or 0.0, 1e-9)
        for i, x in enumerate(xs):
            ax = abs(x)
            cut = self.front_cut if x < 0 else fl
            part = rot(F[i], sgn * a) if x < 0 else F[i]
            if ax >= cut:
                out[i] = part
            elif ax > cut - b:
                t = (ax - (cut - b)) / b
                out[i] = W[i] + t * (part - W[i])
        return out

    def function(self, x):
        if self.front_cut <= abs(x) <= self.back_cut:
            return min(1, max(0, self.__function__(abs(x))))
        else:
            return 1

    def get_3d(self, cell):
        shape_with_bal = cell.basic_cell.midrib(self.y_value, True).data
        shape_wo_bal = cell.basic_cell.midrib(self.y_value, False).data
        return Profile3D(list(self.section(cell.x_values, shape_with_bal, shape_wo_bal)))

    def get_mesh(self, cell):
        """the part as a strip of quads between its upper and lower edge (section of the cell at yvalue,
        stations from front_cut to back_cut, upper and lower station paired by chord fraction)"""
        import numpy as np
        from openglider.mesh import Mesh

        pts = np.asarray(self.get_3d(cell).data)
        xs = np.asarray(cell.x_values)
        up = sorted([(abs(x), i) for i, x in enumerate(xs) if x < 0 and self.front_cut <= abs(x) <= self.back_cut])
        lo = [(abs(x), i) for i, x in enumerate(xs) if x >= 0 and self.front_cut <= abs(x) <= self.back_cut]
        if len(up) < 2 or not lo:
            return Mesh.from_indexed([], {}, {})
        lx = np.array([x for x, _ in lo])
        pairs = [(i, lo[int(np.argmin(np.abs(lx - x)))][1]) for x, i in up]
        vertices = [pts[k] for pair in pairs for k in pair]
        quads = [[2 * k, 2 * k + 2, 2 * k + 3, 2 * k + 1] for k in range(len(pairs) - 1)]
        return Mesh.from_indexed(vertices, {"miniribs": quads}, name=self.name)

    def get_flattened(self, cell):
        prof_3d = self.get_3d(cell)
        prof_flat = prof_3d.flatten()
        prof_normalized = prof_flat.copy().normalize()

        p1 = prof_normalized(-self.back_cut)
        p2 = prof_normalized(-self.front_cut)

        p3 = prof_normalized(self.front_cut)
        p4 = prof_normalized(self.back_cut)
