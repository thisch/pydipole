
all:
	python setup.py build_ext --inplace

test:
	py.test dipole/tests -s
