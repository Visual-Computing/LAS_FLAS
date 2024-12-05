#!/bin/bash

case "$1" in
	r)
		PYTHONPATH="build:." python3 examples/main.py
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
