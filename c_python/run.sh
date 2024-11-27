#!/bin/bash

PYTHONPATH="build:." python3 tests/main.py && gwenview images/image1.png 2>/dev/null
