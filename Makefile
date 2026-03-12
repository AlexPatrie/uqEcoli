PACKAGE_LOCATION=/Users/alexanderpatrie/sms/uqEcoli/uq

.PHONY: documentation
documentation:
	@open "${PACKAGE_LOCATION}/docs/_build/html/index.html"

.PHONY: test
test:
	@uv run pytest ./tests/ -v -s

.PHONY: tutorial4
tutorial4:
	@uv run marimo edit tutorials/04_cell_cycle_and_koopman.py

.PHONY: tutorial-music
tutorial-music:
	@uv run marimo edit tutorials/music.py

.PHONY: tutorial3b
tutorial3b:
	@uv run marimo edit ./tutorials/03b_reactive_sensitivity.py

.PHONY: tutorial3c
tutorial3c:
	@uv run marimo edit ./tutorials/03c_reactive_sensitivity_generalized.py

.PHONY: tutorial5
tutorial5:
	@uv run marimo edit tutorials/05_music_notation.py

.PHONY: tutorial6
tutorial6:
	@uv run marimo edit tutorials/06_calculate_cell_cycle.py

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

.PHONY: biocompose
biocompose:
	@uv run marimo edit ./examples/biocompose.py
