PACKAGE_LOCATION=/Users/alexanderpatrie/sms/uqEcoli/uq

.PHONY: test-sampling
test-sampling:
	@time uv run uq generate-samples \
      api_simulation_default mecillinam test_violacein_with_metabolism \
      --sim-base-path /Users/alexanderpatrie/sms/vEcoli-private/api_integration/sims \
      --cache-dir ./uq_cache \
      --n-samples 20 \
      --live \
      --max-workers 4

.PHONY: add-vecoli
add-vecoli:
	@echo "/Users/alexanderpatrie/sms/vEcoli-private" > $(uv run python -c "import site; print(site.getsitepackages()[0])")/vecoli.pth

.PHONY: install
install:
	@uv lock --no-cache; \
	uv sync --no-cache --all-groups --all-extras \
	make export-deps

.PHONY: export-deps
export-deps:
	@uv pip freeze | sed 's/=.*//' > requirements.txt

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
