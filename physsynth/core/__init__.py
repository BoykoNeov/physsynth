"""Headless DSP core. Pure NumPy/SciPy — no audio I/O, no graphics.

Importing matplotlib (or any plotting/audio library) from this subpackage is a layering violation;
the validation suite asserts it stays clean.
"""
