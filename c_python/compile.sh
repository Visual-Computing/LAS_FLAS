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
	release)
		VERSION="v0.1.7"
		git tag -d "$VERSION"
		git commit -am "$VERSION"
		git tag -a "$VERSION" -m "$VERSION"
		git push
		git push origin --tags
		;;
	*)
		echo "invalid option"
		;;
esac
