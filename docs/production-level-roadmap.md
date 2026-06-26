# Aurelis Production-Level Roadmap

## 1. Current status (verified)

The project is now in a strong engineering baseline:

- CMake configuration succeeds in this environment.
- The project builds successfully.
- The current release validation path passes: 13/13 tests passed with `ctest` and the install tree is generated.
- The release benchmark path is now wired into the CI workflow and uses the same resolved toolchain path as release validation.

This means the codebase is already buildable and functionally validated. It does **not** yet mean the project is fully production-ready.

---

## 2. What production level really means

To be considered production level, Aurelis should satisfy all of the following:

1. Reliable builds on supported platforms
2. Automated tests for correctness, performance, and regression safety
3. Stable APIs and versioned releases
4. Clear deployment and packaging paths
5. Secure, observable, and maintainable runtime behavior
6. Documentation that enables real users to install, run, and extend the system

---

## 3. What is already strong in this repo

The foundation is already promising:

- Modular C/C++/C++ library structure
- Core tensor and math layers implemented
- Logging, config, error handling, and profiling scaffolding present
- Multiple phase-specific libraries are already wired into CMake
- Tests exist for the core pipeline and major subsystems

These are the right building blocks for a production-quality system.

---

## 4. Main gaps that must be closed before production

### A. Release engineering and packaging

Required work:

- Add an install target in CMake
- Produce release builds for Windows/Linux/macOS
- Add versioned package artifacts via CPack
- Provide a simple install procedure for users and CI
- Ensure the public headers and runtime assets are packaged correctly

Why this matters:
- A library is not production-ready if it cannot be installed and reused cleanly.

---

### B. Automated CI/CD

Required work:

- Add GitHub Actions or similar CI workflow
- Run CMake configure, build, and tests on every push/PR
- Add release automation for tagged versions
- Add matrix builds for at least one Windows and one Linux environment

Why this matters:
- Production systems must be reproducible and continuously validated.

---

### C. Test coverage beyond the current smoke suite

Current status:
- The existing suite is good for baseline verification.

Required next tests:

- Unit tests for each major component
- Integration tests for training and inference flows
- Stress tests with long sequences, large tensors, and large vocab sizes
- Memory stress tests and leak detection
- Backward-compatibility tests for checkpoints and saved models
- Fuzz tests for invalid JSON config, bad file input, and malformed data

Why this matters:
- Production systems fail on edge cases, not just happy paths.

---

### D. Performance validation

Required work:

- Benchmark key paths: tensor ops, scan, matmul, lens forward pass, training path
- Compare naive vs BLAS-backed paths
- Profile memory usage and runtime hotspots
- Define acceptable performance baselines for release

Why this matters:
- A production system must be measured, not just compiled.

---

### E. Stability and robustness

Required work:

- Harden file I/O and checkpoint save/load paths
- Improve invalid-input handling across the public APIs
- Add explicit bounds checking for tensor shapes and config dimensions
- Add deterministic failure modes for out-of-memory and resource exhaustion
- Review all `TODO` implementations in the codebase and mark incomplete paths as non-production or finish them

Why this matters:
- Production software must fail safely and predictably.

---

### F. Documentation and developer onboarding

Required work:

- Add installation instructions for Windows/Linux/macOS
- Add a short user guide for using the library and examples
- Add API docs for core classes and modules
- Add contribution guidelines and coding standards
- Add a troubleshooting section for common build and runtime issues

Why this matters:
- A production project must be understandable by other engineers.

---

### G. Security and compliance readiness

Required work:

- Review all file and network input handling
- Avoid unsafe assumptions in config loading and model file parsing
- Add permission and path validation for checkpoint and logging operations
- Add a security review for external dependencies and runtime access

Why this matters:
- Even internal research code should have basic security hygiene before deployment.

---

### H. Observability and operations

Required work:

- Define structured operational logs for training, inference, and failures
- Add runtime metrics and counters
- Add optional profiling/export formats for production dashboards
- Create a standard error reporting and debug path

Why this matters:
- Production systems need to be monitored, debugged, and traced in real deployments.

---

## 5. Recommended production-readiness checklist

Use this as the release gate:

### Build and packaging
- [ ] CMake install target works
- [ ] Release build works on at least one supported platform
- [ ] Packaging artifacts are generated automatically

### Quality gate
- [ ] CI runs on every PR and push
- [ ] All tests pass in CI
- [ ] Performance benchmarks are tracked
- [ ] No known critical TODO paths remain unaddressed

### Reliability
- [ ] Stress tests pass
- [ ] Memory and input validation are hardened
- [ ] Checkpoint compatibility is verified

### Documentation
- [ ] User guide exists
- [ ] API docs exist
- [ ] Example usage is runnable
- [ ] Release notes are documented

### Operations
- [ ] Logging is structured and useful
- [ ] Errors are actionable
- [ ] Profiling/metrics are available for operators

---

## 6. Practical next-step sequence

The fastest realistic path to production level is:

1. Stabilize the build and packaging path
2. Add CI/CD and release automation
3. Expand tests beyond the current baseline
4. Benchmark performance and fix biggest bottlenecks
5. Harden error handling and checkpoint safety
6. Write user-facing documentation and examples
7. Run a formal production-readiness review

---

## 7. Final conclusion

The repository is already in a strong state for a research-to-production transition:

- it builds,
- it tests,
- it has the right architecture and modularity,
- and it already contains important production scaffolding.

However, the project is not yet fully production-level until release packaging, CI/CD, broader test coverage, performance validation, and operational/documentation hardening are completed.

This roadmap gives the exact path to make Aurelis production-ready.
