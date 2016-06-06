# distutils: language = c++
# distutils: sources = dipole/cpp/field.cpp
# distutils: extra_compile_args = [-std=c++11, -fopenmp]
# distutils: extra_link_args = [-lgomp]

from libcpp.vector cimport vector
from libcpp cimport bool
# from libcpp cimport complex as cppcomplex

from numpy import complex128
from numpy cimport complex128_t as complex_t
from numpy cimport float64_t as double_t
import numpy as np
cimport numpy as np
import scipy.constants.constants as co


cdef extern from "cpp/field.hpp":
    cdef vector[vector[double_t]] dipole_radiant_intensity_wrapper(
        vector[vector[double_t]] T,
        vector[vector[double_t]] P,
        vector[vector[double_t]] p,
        vector[vector[double_t]] r,
        vector[double_t] phases,
        double_t k)

    cdef vector[vector[vector[complex_t]]] farfield_dipole_wrapper(
        vector[vector[vector[double_t]]] r,
        vector[vector[double_t]] P,
        vector[vector[double_t]] R,
        vector[double_t] phases,
        double_t k, double_t ts, bool calc_H)

    cdef vector[vector[vector[complex_t]]] general_dipole_wrapper(
        vector[vector[vector[double_t]]] r,
        vector[vector[double_t]] P,
        vector[vector[double_t]] R,
        vector[double_t] phases,
        double_t k, double_t ts, bool calc_H)


def _dipole_ff(np.ndarray[double_t, ndim=3] r,
                np.ndarray[double_t, ndim=2] P,
                np.ndarray[double_t, ndim=2] R,
                np.ndarray[double_t, ndim=1] phases,
                double_t k, double_t t, bool calc_H):

    cdef r0 = r.shape[0]
    cdef r1 = r.shape[1]
    cdef r2 = r.shape[2]

    cdef vector[vector[vector[complex_t]]] aa
    cdef vector[vector[vector[double_t]]] rvec
    cdef vector[vector[double_t]] Pvec
    cdef vector[vector[double_t]] Rvec
    cdef vector[double_t] phases_vec

    rvec.reserve(r0)
    for i in range(r0):
        rvec.push_back(vector[vector[double_t]]())
        rvec[i].reserve(r1)
        for j in range(r1):
            rvec[i].push_back(vector[double_t]())
            rvec[i][j].reserve(r2)
            for kidx in range(r2):
                rvec[i][j].push_back(r[i,j,kidx])

    Pvec.reserve(P.shape[0])
    for i in range(P.shape[0]):
        Pvec.push_back(vector[double_t]())
        Pvec[i].reserve(P.shape[1])
        for j in range(P.shape[1]):
            Pvec[i].push_back(P[i][j])

    Rvec.reserve(R.shape[0])
    for i in range(R.shape[0]):
        Rvec.push_back(vector[double_t]())
        Rvec[i].reserve(R.shape[1])
        for j in range(R.shape[1]):
            Rvec[i].push_back(R[i][j])

    phases_vec.reserve(phases.shape[0])
    for i in range(phases.shape[0]):
        phases_vec.push_back(phases[i])

    aa = farfield_dipole_wrapper(rvec, Pvec, Rvec, phases_vec, k, t, calc_H)

    resvec = np.empty([aa.size(), aa[0].size(), aa[0][0].size()],
                      dtype='complex128')

    for i in range(aa.size()):
        for j in range(aa[i].size()):
            for kidx in range(aa[i][j].size()):
                # print("aa[%d, %d, %d] = %s " % (i, j, kidx, aa[i][j][kidx]))
                resvec[i, j, kidx] = aa[i][j][kidx]
    return resvec


def dipole_e_ff(np.ndarray[double_t, ndim=3] r,
                np.ndarray[double_t, ndim=2] P,
                np.ndarray[double_t, ndim=2] R,
                np.ndarray[double_t, ndim=1] phases,
                double_t k, double_t t):
    return _dipole_ff(r, P, R, phases, k, t, False)


def dipole_h_ff(np.ndarray[double_t, ndim=3] r,
                np.ndarray[double_t, ndim=2] P,
                np.ndarray[double_t, ndim=2] R,
                np.ndarray[double_t, ndim=1] phases,
                double_t k, double_t t):
    return _dipole_ff(r, P, R, phases, k, t, True)


def dipole_radiant_intensity(
        np.ndarray[double_t, ndim=2] T,
        np.ndarray[double_t, ndim=2] P,
        np.ndarray[double_t, ndim=2] p,
        np.ndarray[double_t, ndim=2] r,
        np.ndarray[double_t, ndim=1] phases,
        double_t k):

    cdef r0 = T.shape[0]
    cdef r1 = T.shape[1]

    cdef vector[vector[double_t]] aa
    cdef vector[vector[double_t]] Tvec
    cdef vector[vector[double_t]] Pvec
    cdef vector[vector[double_t]] pvec
    cdef vector[vector[double_t]] rvec
    cdef vector[double_t] phases_vec

    Tvec.reserve(T.shape[0])
    for i in range(T.shape[0]):
        Tvec.push_back(vector[double_t]())
        Tvec[i].reserve(T.shape[1])
        for j in range(T.shape[1]):
            Tvec[i].push_back(T[i][j])

    Pvec.reserve(P.shape[0])
    for i in range(P.shape[0]):
        Pvec.push_back(vector[double_t]())
        Pvec[i].reserve(P.shape[1])
        for j in range(P.shape[1]):
            Pvec[i].push_back(P[i][j])

    pvec.reserve(p.shape[0])
    for i in range(p.shape[0]):
        pvec.push_back(vector[double_t]())
        pvec[i].reserve(p.shape[1])
        for j in range(p.shape[1]):
            pvec[i].push_back(p[i][j])

    rvec.reserve(r.shape[0])
    for i in range(r.shape[0]):
        rvec.push_back(vector[double_t]())
        rvec[i].reserve(r.shape[1])
        for j in range(r.shape[1]):
            rvec[i].push_back(r[i][j])

    phases_vec.reserve(phases.shape[0])
    for i in range(phases.shape[0]):
        phases_vec.push_back(phases[i])

    aa = dipole_radiant_intensity_wrapper(Tvec, Pvec, pvec, rvec, phases_vec, k)
    resvec = np.empty([aa.size(), aa[0].size()], dtype='float64')

    for i in range(aa.size()):
        for j in range(aa[i].size()):
            # print("aa[%d, %d, %d] = %s " % (i, j, kidx, aa[i][j][kidx]))
            resvec[i, j] = aa[i][j]
    return resvec


def dipole_general(np.ndarray[double_t, ndim=3] r,
                   np.ndarray[double_t, ndim=2] P,
                   np.ndarray[double_t, ndim=2] R,
                   np.ndarray[double_t, ndim=1] phases,
                   double_t k, bool poyntingmean=False,
                   bool poyntingstatic=False, double_t t=0):
    cdef r0 = r.shape[0]
    cdef r1 = r.shape[1]
    cdef r2 = r.shape[2]

    cdef vector[vector[vector[complex_t]]] Haa
    cdef vector[vector[vector[complex_t]]] Eaa
    cdef vector[vector[vector[double_t]]] rvec
    cdef vector[vector[double_t]] Pvec
    cdef vector[vector[double_t]] Rvec
    cdef vector[double_t] phases_vec

    # better way to initialize all my vectors??
    rvec.reserve(r0)
    for i in range(r0):
        rvec.push_back(vector[vector[double_t]]())
        rvec[i].reserve(r1)
        for j in range(r1):
            rvec[i].push_back(vector[double_t]())
            rvec[i][j].reserve(r2)
            for kidx in range(r2):
                rvec[i][j].push_back(r[i,j,kidx])

    Pvec.reserve(P.shape[0])
    for i in range(P.shape[0]):
        Pvec.push_back(vector[double_t]())
        Pvec[i].reserve(P.shape[1])
        for j in range(P.shape[1]):
            Pvec[i].push_back(P[i][j])

    Rvec.reserve(R.shape[0])
    for i in range(R.shape[0]):
        Rvec.push_back(vector[double_t]())
        Rvec[i].reserve(R.shape[1])
        for j in range(R.shape[1]):
            Rvec[i].push_back(R[i][j])

    phases_vec.reserve(phases.shape[0])
    for i in range(phases.shape[0]):
        phases_vec.push_back(phases[i])

    Eaa = general_dipole_wrapper(rvec, Pvec, Rvec, phases_vec, k, t, False)
    Haa = general_dipole_wrapper(rvec, Pvec, Rvec, phases_vec, k, t, True)

    Eresvec = np.empty([Eaa.size(), Eaa[0].size(), Eaa[0][0].size()], dtype='complex128')
    Hresvec = np.empty_like(Eresvec)

    for i in range(Eaa.size()):
        for j in range(Eaa[i].size()):
            for kidx in range(Eaa[i][j].size()):
                # print("aa[%d, %d, %d] = %s " % (i, j, kidx, aa[i][j][kidx]))
                Eresvec[i, j, kidx] = Eaa[i][j][kidx]
                Hresvec[i, j, kidx] = Haa[i][j][kidx]
    if poyntingmean:
        # here we compute the time average of the poynting vector for
        # time-harmonic fields. (see wikipedia poynting vec)
        return 0.5*np.cross(Eresvec, Hresvec.conjugate()).real
    elif poyntingstatic:
        # this is the expression for the poynting vector at time t assuming
        # time-harmonic fields. (see wikipedia poynting vec)
        return (0.5*np.cross(Eresvec,
                             Hresvec.conjugate() + Hresvec*np.exp(2j*k*co.c*t)).real)
    else:
        return Eresvec, Hresvec
