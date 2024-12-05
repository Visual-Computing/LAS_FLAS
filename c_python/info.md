## How to publish a new version
- Run `git checkout main` and `git pull` to be sure, all updates are fetched
- Edit version number in `c_python/src/vc_flas/__init__.py` to `x.y.z`
- Run `git add -A`, `git commit -m 'vx.y.z'` and `git tag -a vx.y.z -m 'vx.y.z'`
- Run `git push` and `git push origin --tags`
