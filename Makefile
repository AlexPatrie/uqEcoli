PACKAGE_LOCATION=/Users/alexanderpatrie/sms/uqEcoli/uq

.PHONY: install
install:
	@uv lock --no-cache; \
	uv sync --no-cache --all-groups --all-extras \
	make export-deps

.PHONY: export-deps
export-deps:
	@uv pip freeze | sed 's/=.*//' > requirements.txt

.PHONY: show-docs
show-docs:
	@open "${PACKAGE_LOCATION}/docs/_build/html/index.html"

.PHONY: test
test:
	@uv run pytest ./tests/ -v -s

.PHONY: check
check: ## Run code quality tools.
	@echo "🚀 Checking lock file consistency with 'pyproject.toml'"
	@uv lock --locked
	@echo "🚀 Linting code: Running pre-commit"
	@uv run pre-commit run -a
	@echo "🚀 Static type checking: Running mypy"
	@uv run mypy
	@echo "🚀 Checking for obsolete dependencies: Running deptry"
	@uv run deptry .

.PHONY: docs
docs: ## Build Sphinx HTML docs
	@uv run sphinx-build -b html docs docs/_build/html
	@echo "Docs built at docs/_build/html/index.html"

.PHONY: commits
commits:
	@./commits.sh