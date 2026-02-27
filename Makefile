PYTHON ?= .venv/bin/python

.PHONY: help test test-py test-rust parity bench demo fmt fmt-check rust-interactive rust-interactive-dense rust-interactive-dense-parallel perf-note legacy-test-py legacy-bench legacy-demo

help:
	@echo "Targets:"
	@echo "  make test       - run Rust core tests + Python<->Rust parity test"
	@echo "  make test-py    - run parity test suite"
	@echo "  make test-rust  - run Rust core tests"
	@echo "  make parity     - run cross-language parity test"
	@echo "  make bench      - run dense interactive Rust scene (parallel)"
	@echo "  make demo       - run Rust interactive demo scene"
	@echo "  make fmt        - format Rust + Python code (black if installed)"
	@echo "  make fmt-check  - check Rust/Python formatting (black if installed)"
	@echo "  make rust-interactive - run Rust interactive light-control demo"
	@echo "  make rust-interactive-dense - run dense Rust interactive stress scene"
	@echo "  make rust-interactive-dense-parallel - run dense scene with parallel renderer"
	@echo "  make perf-note  - append a timestamped perf log template entry"
	@echo "  make legacy-test-py - run archived Python renderer tests"
	@echo "  make legacy-bench   - run archived Python benchmark"
	@echo "  make legacy-demo    - run archived Python demo (headless)"

test: test-py test-rust

test-py:
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py' -v

test-rust:
	cd rust && cargo test -p fast_spheres_core

parity:
	$(PYTHON) -m unittest tests.test_rust_parity -v

bench:
	$(MAKE) rust-interactive-dense-parallel

demo:
	$(MAKE) rust-interactive

rust-interactive:
	cd rust && cargo run -p fast_spheres_app -- scenes/demo_scene.json --interactive --scene2 scenes/demo_scene_alt.json

rust-interactive-dense:
	cd rust && cargo run -p fast_spheres_app -- scenes/demo_dense.json --interactive --continuous --scene2 scenes/demo_dense_parallel.json

rust-interactive-dense-parallel:
	cd rust && cargo run -p fast_spheres_app -- scenes/demo_dense_parallel.json --interactive --continuous --scene2 scenes/demo_dense.json

perf-note:
	@ts=$$(date -u +"%Y-%m-%d %H:%M:%S UTC"); \
	echo "| $$ts | <machine> | demo_scene | on-change | <fps> | |" >> PERF_NOTES.md; \
	echo "| $$ts | <machine> | demo_scene | continuous | <fps> | |" >> PERF_NOTES.md; \
	echo "| $$ts | <machine> | demo_dense | on-change | <fps> | |" >> PERF_NOTES.md; \
	echo "| $$ts | <machine> | demo_dense | continuous | <fps> | |" >> PERF_NOTES.md; \
	echo "Appended template rows to PERF_NOTES.md"

fmt:
	cd rust && cargo fmt --all
	@if $(PYTHON) -m black --version >/dev/null 2>&1; then \
		$(PYTHON) -m black tests archive/python/src archive/python/tests; \
	else \
		echo "black not installed in $(PYTHON); skipping Python formatting"; \
	fi

fmt-check:
	cd rust && cargo fmt --all -- --check
	@if $(PYTHON) -m black --version >/dev/null 2>&1; then \
		$(PYTHON) -m black --check tests archive/python/src archive/python/tests; \
	else \
		echo "black not installed in $(PYTHON); skipping Python format check"; \
	fi

legacy-test-py:
	PYTHONPATH=archive/python $(PYTHON) -m unittest discover -s archive/python/tests -p 'test_*.py' -v

legacy-bench:
	PYTHONPATH=archive/python $(PYTHON) -m src.cli benchmark --preset quick --method auto

legacy-demo:
	PYTHONPATH=archive/python MPLBACKEND=Agg $(PYTHON) -m src.cli demo --method auto --no-show
