electric dipole radiation in near and far field
================================

# Compilation

    python setup.py build_ext -i

# Run unit tests

```
py.test
py.test --interactive  # unskip interactive tests (matplotlib plots are shown)
py.test --nocapturelog -s  # shows the log output
py.test 'dipole/tests/test_ring.py::TestRing::test_rolf_pishift[True]' --interactive  # run a single test
```