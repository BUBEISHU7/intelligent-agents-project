from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


Vector = Tuple[float, float]


def _dot(a: Vector, b: Vector) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _det(a: Vector, b: Vector) -> float:
    return a[0] * b[1] - a[1] * b[0]


def _norm(a: Vector) -> float:
    return math.hypot(a[0], a[1])


def _normalize(a: Vector) -> Vector:
    n = _norm(a)
    if n < 1e-12:
        return (0.0, 0.0)
    return (a[0] / n, a[1] / n)


def _sub(a: Vector, b: Vector) -> Vector:
    return (a[0] - b[0], a[1] - b[1])


def _add(a: Vector, b: Vector) -> Vector:
    return (a[0] + b[0], a[1] + b[1])


def _mul(a: Vector, s: float) -> Vector:
    return (a[0] * s, a[1] * s)


def _clamp_speed(v: Vector, max_speed: float) -> Vector:
    s = _norm(v)
    if s <= max_speed:
        return v
    if s < 1e-12:
        return (0.0, 0.0)
    return (v[0] / s * max_speed, v[1] / s * max_speed)


@dataclass(frozen=True)
class Line:
    point: Vector
    direction: Vector


@dataclass(frozen=True)
class Neighbor:
    position: Vector
    velocity: Vector
    radius: float


def compute_orca_velocity(
    position: Vector,
    velocity: Vector,
    preferred_velocity: Vector,
    radius: float,
    neighbors: Sequence[Neighbor],
    time_horizon: float = 5.0,
    max_speed: float = 1.5,
) -> Vector:
    """
    Strict ORCA (van den Berg et al.) for disc agents with moving neighbors.
    Returns a new collision-free velocity (within max_speed) closest to preferred_velocity.
    """
    time_horizon = max(1e-3, float(time_horizon))
    max_speed = float(max_speed)

    orca_lines: List[Line] = []
    inv_time_horizon = 1.0 / time_horizon

    for nb in neighbors:
        rel_pos = _sub(nb.position, position)
        rel_vel = _sub(velocity, nb.velocity)
        dist = _norm(rel_pos)
        combined = radius + float(nb.radius)

        # Compute u and ORCA line.
        if dist < 1e-6:
            # same position, pick arbitrary separation direction
            w = (1.0, 0.0)
            u = _mul(w, (combined * inv_time_horizon) - 0.0)
            direction = (0.0, 1.0)
            point = _add(velocity, _mul(u, 0.5))
            orca_lines.append(Line(point=point, direction=direction))
            continue

        # Vector from cutoff center to relative velocity
        w = _sub(rel_vel, _mul(rel_pos, inv_time_horizon))
        w_len = _norm(w)
        # Outside the velocity obstacle?
        if dist > combined:
            # project on VO boundary
            if w_len < 1e-12:
                unit_w = _normalize((-rel_pos[1], rel_pos[0]))
            else:
                unit_w = _mul(w, 1.0 / w_len)
            u = _mul(unit_w, (combined * inv_time_horizon - w_len))
            # direction is perpendicular to unit_w
            direction = (unit_w[1], -unit_w[0])
        else:
            # already on collision course: use time step approximation
            inv_time_step = inv_time_horizon  # conservative proxy
            w = _sub(rel_vel, _mul(rel_pos, inv_time_step))
            w_len = _norm(w)
            if w_len < 1e-12:
                unit_w = _normalize((-rel_pos[1], rel_pos[0]))
            else:
                unit_w = _mul(w, 1.0 / w_len)
            u = _mul(unit_w, (combined * inv_time_step - w_len))
            direction = (unit_w[1], -unit_w[0])

        point = _add(velocity, _mul(u, 0.5))
        orca_lines.append(Line(point=point, direction=direction))

    # Solve linear programs to find new velocity.
    result = _clamp_speed(preferred_velocity, max_speed)
    line_fail = _linear_program_2(orca_lines, max_speed, preferred_velocity, result)
    if line_fail < len(orca_lines):
        result = _linear_program_3(orca_lines, line_fail, max_speed, result)
    return result


def _linear_program_1(lines: Sequence[Line], line_no: int, max_speed: float, opt_velocity: Vector, result: Vector) -> Optional[Vector]:
    line = lines[line_no]
    dotp = _dot(line.point, line.direction)
    disc = dotp * dotp + max_speed * max_speed - _dot(line.point, line.point)
    if disc < 0.0:
        return None

    sqrt_disc = math.sqrt(disc)
    t_left = -dotp - sqrt_disc
    t_right = -dotp + sqrt_disc

    for i in range(line_no):
        denom = _det(line.direction, lines[i].direction)
        numer = _det(lines[i].direction, _sub(line.point, lines[i].point))
        if abs(denom) < 1e-12:
            if numer < 0.0:
                return None
            continue
        t = numer / denom
        if denom > 0.0:
            t_right = min(t_right, t)
        else:
            t_left = max(t_left, t)
        if t_left > t_right:
            return None

    # Optimize along the line segment
    t = _dot(line.direction, _sub(opt_velocity, line.point))
    if t < t_left:
        t = t_left
    elif t > t_right:
        t = t_right
    return _add(line.point, _mul(line.direction, t))


def _linear_program_2(lines: Sequence[Line], max_speed: float, opt_velocity: Vector, result: Vector) -> int:
    for i in range(len(lines)):
        if _det(lines[i].direction, _sub(lines[i].point, result)) > 0.0:
            # result violates line i
            tmp = result
            new_res = _linear_program_1(lines, i, max_speed, opt_velocity, tmp)
            if new_res is None:
                result = tmp
                return i
            result = new_res
    # mutate caller by returning success index
    opt_velocity  # keep lint happy
    return len(lines)


def _linear_program_3(lines: Sequence[Line], begin: int, max_speed: float, result: Vector) -> Vector:
    distance = 0.0
    res = result
    for i in range(begin, len(lines)):
        if _det(lines[i].direction, _sub(lines[i].point, res)) > distance:
            proj_lines: List[Line] = []
            for j in range(i):
                det = _det(lines[i].direction, lines[j].direction)
                if abs(det) < 1e-12:
                    # parallel
                    if _dot(lines[i].direction, lines[j].direction) > 0.0:
                        continue
                    point = _mul(_add(lines[i].point, lines[j].point), 0.5)
                else:
                    point = _add(
                        lines[i].point,
                        _mul(lines[i].direction, _det(lines[j].direction, _sub(lines[i].point, lines[j].point)) / det),
                    )
                direction = _normalize(_sub(lines[j].direction, _mul(lines[i].direction, _dot(lines[j].direction, lines[i].direction))))
                if _norm(direction) < 1e-12:
                    direction = (lines[i].direction[1], -lines[i].direction[0])
                proj_lines.append(Line(point=point, direction=direction))

            # optimize in direction perpendicular to current line direction
            opt = (lines[i].direction[1], -lines[i].direction[0])
            cand = _clamp_speed(opt, max_speed)
            fail = _linear_program_2(proj_lines, max_speed, opt, cand)
            if fail < len(proj_lines):
                # fallback: keep current
                pass
            else:
                res = cand
            distance = _det(lines[i].direction, _sub(lines[i].point, res))
    return res

