"""Root conftest.

Its mere presence puts the repository root on ``sys.path`` (pytest prepend import mode), so
``import physsynth`` works without an editable install. Test files additionally import the
sibling ``tests/helpers.py`` module (the ``tests`` directory is added to the path for each test).
"""
