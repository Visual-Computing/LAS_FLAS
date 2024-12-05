#!/bin/bash

case "$1" in
	r)
		PYTHONPATH="build:." python3 examples/main.py && gwenview images/image1.png 2>/dev/null
		;;
	e)
		PYTHONPATH="build:." python3 examples/metrics.py
		;;
	t)
		PYTHONPATH="$PWD" pytest
		;;
	*)
		echo "invalid option"
		;;
esac
