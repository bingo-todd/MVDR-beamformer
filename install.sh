#!/bin/sh

python setup.py sdist
pip install dist/*.tar.gz
