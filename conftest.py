def setup_logger(nocol=False):
    import logging
    import sys

    for lname in ['dip']:
        LG = logging.getLogger(lname)
        try:
            from falafel.logger import Formatter
            from falafel.logger import NOCOLORS
            LG.setLevel('DEBUG')
            sh = logging.StreamHandler(sys.stdout)

            if nocol:
                sh.setFormatter(Formatter(pre=NOCOLORS, lenstrip=None, contline=None))
            else:
                sh.setFormatter(Formatter())
            LG.addHandler(sh)
        except ImportError:
            logging.basicConfig(level='DEBUG', stream=sys.stdout)

setup_logger()


def pytest_addoption(parser):
    parser.addoption("--interactive", action="store_true")
    parser.addoption("--log", action="store_true")
    parser.addoption("--logdir", default='log')
