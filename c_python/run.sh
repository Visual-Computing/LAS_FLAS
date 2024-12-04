#!/bin/bash

case "$1" in
	r)
		PYTHONPATH="build:." python3 examples/main.py && gwenview images/image1.png 2>/dev/null
		;;
	t)
		PYTHONPATH="$PWD" pytest
		;;
	*)
		echo "invalid option"
		;;
esac
