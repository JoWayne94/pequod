"""Optional fused CPU kernels; float64, no fast-math or changed timesteps.

Set PEQUOD_QG_NUMBA=0 to exercise the equivalent NumPy reference path.
JIT compilation is cached on disk and paid only on the first invocation.
"""
import os

import numpy as np

try:
    from numba import config, njit
except ImportError:
    AVAILABLE = False

    def njit(**kwargs):
        return lambda function: function
else:
    AVAILABLE = not config.DISABLE_JIT and os.environ.get("PEQUOD_QG_NUMBA", "1") != "0"


@njit(cache=True)
def divergence(fx, fy, dx, dy, dt, cons, qx, qy, q):
    ny, nx = cons.shape
    for j in range(ny):
        jm = j-1 if j else ny-1
        for i in range(nx):
            im = i-1 if i else nx-1
            x = (fx[j, im]-fx[j, i])/dx
            y = (fy[jm, i]-fy[j, i])/dy
            qx[j, i], qy[j, i] = x, y
            q[j, i] = x+y
            cons[j, i] += .5*dt*(x+y)


@njit(cache=True)
def predictor_bounds(ax, ay, cons, qx, qy, dt, Mx, mx, My, my):
    ny, nx = cons.shape
    for j in range(ny):
        jm = j-1 if j else ny-1
        for i in range(nx):
            im = i-1 if i else nx-1
            c = cons[j, i]
            Mx[j, i] = np.maximum(np.maximum(ax[j, i], c), ax[j, im])+dt*qy[j, i]
            mx[j, i] = np.minimum(np.minimum(ax[j, i], c), ax[j, im])+dt*qy[j, i]
            My[j, i] = np.maximum(np.maximum(ay[j, i], c), ay[jm, i])+dt*qx[j, i]
            my[j, i] = np.minimum(np.minimum(ay[j, i], c), ay[jm, i])+dt*qx[j, i]


@njit(cache=True)
def extrapolate(ax0, ax1, ay0, ay1, cons, Mx, mx, My, my, workx, worky):
    ny, nx = cons.shape
    eps = np.finfo(np.float64).eps
    for j in range(ny):
        jm = j-1 if j else ny-1
        for i in range(nx):
            im = i-1 if i else nx-1
            c = 2*cons[j, i]
            ax1[j, i] = np.minimum(np.maximum(c-ax0[j, im], mx[j, i]+eps), Mx[j, i]-eps)
            ay1[j, i] = np.minimum(np.maximum(c-ay0[jm, i], my[j, i]+eps), My[j, i]-eps)
            workx[j, i] = c-ax0[j, i]
            worky[j, i] = c-ay0[j, i]
    for j in range(ny):
        jp = j+1 if j+1 < ny else 0
        for i in range(nx):
            ip = i+1 if i+1 < nx else 0
            ax0[j, i] = np.minimum(np.maximum(workx[j, ip], mx[j, ip]+eps), Mx[j, ip]-eps)
            ay0[j, i] = np.minimum(np.maximum(worky[jp, i], my[jp, i]+eps), My[jp, i]-eps)


@njit(cache=True)
def vertices(p, out, mode, north_wall):
    ny, nx = p.shape
    for j in range(ny):
        jp = j+1 if j+1 < ny else 0
        for i in range(nx):
            ip = i+1 if i+1 < nx else 0
            north, east, northeast = p[jp, i], p[j, ip], p[jp, ip]
            if mode == 1 and j == ny-1:  # Channel
                north = northeast = north_wall
            elif mode == 2:  # Basin
                if j == ny-1:
                    north = northeast = 0.
                if i == nx-1:
                    east = northeast = 0.
            out[j, i] = .25*((p[j, i]+north)+east+northeast)


@njit(cache=True)
def velocity(p, u, v, previous_u, previous_v, cu, cv, dx, dy, alpha, mode, south_u, west_v):
    ny, nx = p.shape
    for j in range(ny):
        jm = j-1 if j else ny-1
        for i in range(nx):
            im = i-1 if i else nx-1
            raw_u = (p[jm, i]-p[j, i])/dy if j or mode == 0 else south_u
            raw_v = (p[j, i]-p[j, im])/dx if i or mode != 2 else west_v
            u[j, i] = (1+alpha)*raw_u-alpha*previous_u[j, i]
            v[j, i] = (1+alpha)*raw_v-alpha*previous_v[j, i]
            previous_u[j, i], previous_v[j, i] = raw_u, raw_v
    for j in range(ny):
        jm = j-1 if j else ny-1
        for i in range(nx):
            im = i-1 if i else nx-1
            cu[j, i] = .5*(u[j, im]+u[j, i]) if i or mode != 2 else 0.
            cv[j, i] = .5*(v[jm, i]+v[j, i]) if j or mode == 0 else 0.


@njit(cache=True)
def limiter_sources(ax, ay, cu, cv, q, forcing, dx, dy, dt, Mx, mx, My, my):
    ny, nx = ax.shape
    for j in range(ny):
        jm = j-1 if j else ny-1
        for i in range(nx):
            im = i-1 if i else nx-1
            f = (ax[j, i]-ax[j, im])/dx
            g = (ay[j, i]-ay[jm, i])/dy
            ix = dt*(q[j, i]+cu[j, i]*f+forcing[j, i])
            iy = dt*(q[j, i]+cv[j, i]*g+forcing[j, i])
            Mx[j, i] += ix
            mx[j, i] += ix
            My[j, i] += iy
            my[j, i] += iy


@njit(cache=True)
def diffusion(u, out, dx2, dy2, dt_nu):
    ny, nx = out.shape
    for j in range(ny):
        for i in range(nx):
            c = 2*u[j+1, i+1]
            y = (u[j+2, i+1]-c+u[j, i+1])/dy2
            x = (u[j+1, i+2]-c+u[j+1, i])/dx2
            out[j, i] = (y+x)*dt_nu
