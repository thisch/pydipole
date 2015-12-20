electric dipole radiation in near and far field
================================

The electric and the magnetic field of a single radiating dipole in vacuum centered at the origin (r=0) are given by

![](https://upload.wikimedia.org/math/7/b/4/7b487096b3b9661fd46a5768a8a36407.png)
![](https://upload.wikimedia.org/math/0/5/4/054a31e26998ea459e680f2788fbf692.png)

The function ``dipole.field.dipole_general`` evaluates the electric and/or magnetic field of a set of oscillating dipoles at specified positions. If only the far field is of interest, the optimized function ``dipole.field.dipole_e_ff`` can be used.
In the far field limit the fields are given by 

![](https://upload.wikimedia.org/math/1/b/a/1ba94136987feca2fdd4067a9a3cd20f.png)
![](https://upload.wikimedia.org/math/0/6/d/06d634484563b8c4c576ee1cca59fb46.png)

See the unit tests in ``dipole/tests/*.py`` for examples on how to use the mentioned functions.

# API
```
def dipole_general(np.ndarray[double_t, ndim=3] r,
                   np.ndarray[double_t, ndim=2] P,
                   np.ndarray[double_t, ndim=2] R,
                   np.ndarray[double_t, ndim=1] phases,
                   double_t k, bool poyntingmean=False,
                   bool poyntingstatic=False, double_t t=0):
    ...

def dipole_e_ff(np.ndarray[double_t, ndim=3] r,
                np.ndarray[double_t, ndim=2] P,
                np.ndarray[double_t, ndim=2] R,
                np.ndarray[double_t, ndim=1] phases,
                double_t k, double_t t=0):
    ...
    
```


# Compilation

    python setup.py build_ext -i

# Run unit tests

```
py.test
py.test --interactive  # unskip interactive tests (matplotlib plots are shown)
py.test --nocapturelog -s  # shows the log output
py.test 'dipole/tests/test_ring.py::TestRing::test_rolf_pishift[True]' --interactive  # run a single test
```
