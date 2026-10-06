# Adding a method

1. Choose a stable lowercase method ID.
2. Create `methods/<id>/` with README, license, notices, dependencies, configs,
   scripts, results, and tests.
3. Register the method and any new datasets or checkpoints under `registry/`.
4. Make paths configurable; never commit workstation paths or credentials.
5. Record preprocessing, split, color space, quantization, inference geometry,
   metrics, checkpoint hash, and upstream revision for every result.
6. Run `python tools/validate_registry.py` and the method smoke tests.
