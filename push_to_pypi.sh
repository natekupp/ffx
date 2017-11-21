# they don't recommend using "setup.py register upload" anymore
python setup.py sdist bdist_egg bdist_wheel # just build
twine upload dist/* # pip install twine if needed.
