#!/bin/bash

case "$1" in
	c)
		if [ ! -d build ]; then
			mkdir build
		fi

		cmake -B build -DCMAKE_BUILD_TYPE=Release -G "Unix Makefiles"

		cd build
		make -j 10
		;;
	p)
		pip install .
		;;
	ci)
		cibuildwheel --output-dir "dist"
		;;
	*)
		echo "invalid option"
		;;
esac
