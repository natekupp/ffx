pylint:
	pylint -j 0 `git ls-files '*.py'` --rcfile=.pylintrc

black:
	black ffx --line-length 100 --target-version py27 --target-version py35 --target-version py36 --target-version py37 --target-version py38 -S --fast --exclude "build/|buck-out/|dist/|_build/|\.eggs/|\.git/|\.hg/|\.mypy_cache/|\.nox/|\.tox/|\.venv/"

isort:
	isort -rc

validate: pylint isort black

pypi:
	rm dist/*
	python setup.py sdist bdist_egg bdist_wheel
	twine upload dist/*
	#twine upload --repository-url https://test.pypi.org/legacy/ dist/* # testpypi
