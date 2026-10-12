# Contributing

Keep changes focused and add tests for scientific invariants: split independence, coordinate
transforms, labels, failed-prediction denominators and checkpoint recovery. Run Ruff and unittest.

Experiment changes require a separate YAML and validation output. Record software, config,
seed, source manifest/hash and model checkpoint/hash. Do not reuse output directories across
experiments or infer population identity from old notebook variable names.

Do not commit datasets, model weights, private evidence or access credentials. Raw evidence is
ignored by Git. Keep synthetic test outputs out of research metrics and label hardware validation gaps.
