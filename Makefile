PYTHON ?= .venv/bin/python

.PHONY: help test test-py test-rust parity bench demo fmt fmt-check

help:
	@echo "Targets:"
	@echo "  make test       - run Python + Rust tests"
	@echo "  make test-py    - run Python test suite"
	@echo "  make test-rust  - run Rust core tests"
	@echo "  make parity     - run cross-language parity test"
	@echo "  make bench      - run quick Python benchmark"
	@echo "  make demo       - run Python demo (headless)"
	@echo "  make fmt        - format Rust + Python code (black if installed)"
	@echo "  make fmt-check  - check Rust/Python formatting (black if installed)"

test: test-py test-rust

test-py:
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py' -v

test-rust:
	cd rust && cargo test -p fast_spheres_core

parity:
	$(PYTHON) -m unittest tests.test_rust_parity -v

bench:
	$(PYTHON) -m src.cli benchmark --preset quick --method auto

demo:
	MPLBACKEND=Agg $(PYTHON) -m src.cli demo --method auto --no-show

fmt:
	cd rust && cargo fmt --all
	@if $(PYTHON) -m black --version >/dev/null 2>&1; then \
		$(PYTHON) -m black src tests; \
	else \
		echo "black not installed in $(PYTHON); skipping Python formatting"; \
	fi

fmt-check:
	cd rust && cargo fmt --all -- --check
	@if $(PYTHON) -m black --version >/dev/null 2>&1; then \
		$(PYTHON) -m black --check src tests; \
	else \
		echo "black not installed in $(PYTHON); skipping Python format check"; \
	fi
