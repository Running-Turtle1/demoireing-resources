# Contributing

Add each method under `methods/<method-id>/` and each dataset description under
`datasets/<dataset-id>/`. Register new resources in `registry/` and run:

```bash
python tools/validate_registry.py
```

A method contribution must include attribution and license information,
portable commands, at least one smoke test, structured result metadata, and
checkpoint hashes when weights are published. Do not commit datasets, model
weights, credentials, absolute workstation paths, or raw training logs.
