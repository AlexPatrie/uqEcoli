PACKAGE_LOCATION=/Users/alexanderpatrie/sms/uqEcoli/uq

.PHONY: documentation
documentation:
	@open "${PACKAGE_LOCATION}/docs/_build/html/index.html"

.PHONY: test
test:
	@uv run pytest uq/tests/ -v -s