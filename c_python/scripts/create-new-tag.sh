#!/bin/bash

set -e

if [ -z "$1" ]; then
	echo "first argument should contain version number"
	exit 1
fi

git add -A
git commit -m "$1"
git tag -a "$1" -m "$1"
git push
git push origin --tags

