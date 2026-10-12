# Data provenance and audit boundaries

SpineLocus operates on locally supplied Digitally Reconstructed Radiographs (DRRs) with thoracic landmark annotations. Neither the underlying CT volumes nor the derived images are distributed here.

## Local archive audit (12 October 2026)

| Inventory | Images | Distinct subject identifiers |
| --- | ---: | ---: |
| Source DRR directory | 4,553 JPEG | 80 |
| Curated experiment dataset | 3,362 | 74 |
| Training | 2,022 | 44 |
| Validation | 750 | 15 |
| Held-out test | 590 | 15 |

A filename-based local audit matched **3,362 of 3,362 selected image basenames** to the source DRR directory and found **zero duplicated basenames** and **zero subject identifiers shared among train, validation and test**. The remaining source directory contains **1,191 unselected images**, including images from **six subject identifiers absent** from the curated subset.

Dataset curation aimed to keep a consistent T1–T11 landmark task; views with additional vertebral levels were not uniformly available across subjects. The reason for excluding each individual image was not separately audited. The effect of this selection on generalization has not been measured.

**Limits:** File-naming conventions are consistent with COPDGene-style subject identifiers but do not by themselves prove CT origin or the terms under which the scans were accessed. Basename correspondence does not prove byte-for-byte or pixel-for-pixel identity. No dataset, subject-level manifest, label, CT, DRR or checkpoint is released. Dataset-access and derivative-redistribution rights require independent verification. The held-out test set has not been used to substantiate the public validation metrics.

## Scientific reproducibility

The CLI includes manifest and patient-level split checks, duplicate-data detection and experimental audit commands. Public unit tests use synthetic data. Reproduction of the reported validation metrics requires properly authorized access to the relevant source data and model checkpoints, along with the matching protocol and software versions.
