#!/bin/bash

case "$1" in
	r)
		PYTHONPATH="build:." python3 tests/main.py && gwenview images/image1.png 2>/dev/null
		;;
	t)
		PYTHONPATH="build:." python3 tests/test.py
		;;
	*)
		echo "invalid option"
		;;
esac
