[![Build Status](https://travis-ci.org/thisch/pydipole.svg?branch=master)](http://travis-ci.org/thisch/pydipole)

Electric dipole radiation in near and far field
===============================================

The electric and the magnetic field of a set of non-interacting radiating
electric dipoles in vacuum are given by

![](https://github.com/thisch/pydipole/raw/master/doc/equations-0.png)
![](https://github.com/thisch/pydipole/raw/master/doc/equations-2.png)

where `p_n` are the dipole moments and `varphi_n` are the temporal phases of
the oscillating dipoles. The quantities `r'` and `rvechat'` depend on the
positions of the dipoles `a_n`,

![](https://github.com/thisch/pydipole/raw/master/doc/equations-1.png)

The function ``dipole.field.dipole_general`` evaluates the electric and/or
magnetic field of a set of oscillating dipoles at specified observation
points. If only the far field is of interest, the optimized function
``dipole.field.dipole_e_ff`` can be used.  In the far field limit the fields
are given by

![](https://github.com/thisch/pydipole/raw/master/doc/equations-3.png)
![](https://github.com/thisch/pydipole/raw/master/doc/equations-4.png)

``dipole.field.dipole_radiant_intensity`` computes the average power
radiated and is given by ([radiant intensity](https://en.wikipedia.org/wiki/Radiant_intensity))

![](https://github.com/thisch/pydipole/raw/master/doc/equations-5.png)

See the examples in `examples/*.py` and the unit tests in `dipole/tests/*.py` for examples on how to use the mentioned functions.

## API
```
def dipole_radiant_intensity(
        np.ndarray[double_t, ndim=2] T,  # theta coords (observation points)
        np.ndarray[double_t, ndim=2] P,  # phi coords (observation points)
        np.ndarray[double_t, ndim=2] p,  # dipole moments
        np.ndarray[double_t, ndim=2] r,  # dipole positions
        np.ndarray[double_t, ndim=1] phases,
        double_t k):
    ...

def dipole_general(np.ndarray[double_t, ndim=3] r, # observation points
                   np.ndarray[double_t, ndim=2] P, # dipole moments
                   np.ndarray[double_t, ndim=2] R, # dipole positions
                   np.ndarray[double_t, ndim=1] phases,
                   double_t k, # wave number
                   bool poyntingmean=False, # TODO deprecate this kwarg
                   bool poyntingstatic=False, # TODO deprecate this kwarg
                   double_t t=0):
    ...

# computes E field in the far-field region
def dipole_e_ff(np.ndarray[double_t, ndim=3] r,
                np.ndarray[double_t, ndim=2] P,
                np.ndarray[double_t, ndim=2] R,
                np.ndarray[double_t, ndim=1] phases,
                double_t k, double_t t=0):
    ...

# computes H field in the far-field region
def dipole_h_ff(np.ndarray[double_t, ndim=3] r,
                np.ndarray[double_t, ndim=2] P,
                np.ndarray[double_t, ndim=2] R,
                np.ndarray[double_t, ndim=1] phases,
                double_t k, double_t t=0):
    ...

```

## References

[Notes by Alpar Sevgen](http://www.phys.boun.edu.tr/~sevgena/p202/docs/Electric%20dipole%20radiation.pdf)
[Electric Dipole Radiation (1)](https://en.wikipedia.org/wiki/Multipole_radiation#Electric_dipole_radiation)
[Electric Dipole Radiation (2)](https://en.wikipedia.org/wiki/Dipole#Dipole_radiation)

## Requirements
* Python 3
* Numpy
* Cython
* Matplotlib
* py.test 

## Compilation

    python setup.py build_ext -i

## Run unit tests

```
py.test
py.test --interactive  # unskip interactive tests (matplotlib plots are shown)
py.test --nocapturelog -s  # shows the log output
py.test 'dipole/tests/test_ring.py::TestRing::test_rolf_pishift[True]' --interactive  # run a single test
```
