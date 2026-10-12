# Third-party software, methods and data

SpineLocus is an independently implemented software project authored and maintained by **Luca Serioli**. It uses established external software packages and published model architectures. These are acknowledged for their respective contributions; such acknowledgments do not imply co-authorship of the SpineLocus implementation.

- **Ultralytics / YOLO-Pose:** https://github.com/ultralytics/ultralytics — upstream package under AGPL-3.0.
- **timm / PyTorch Image Models:** https://github.com/huggingface/pytorch-image-models — upstream package under Apache-2.0.
- **PyTorch and torchvision:** https://github.com/pytorch — upstream project licensing applies to each distribution.
- **HRNet architecture and related research:** https://github.com/HRNet/HRNet-Human-Pose-Estimation — original architecture and publications are separate from the SpineLocus implementation.

No third-party library source, pretrained weights, CT volumes, clinical images, DRRs or individual-level annotations are bundled in this repository. Their access, licenses, research-use conditions and downstream redistribution rights must be assessed independently. File-naming conventions are not evidence of permission to distribute the underlying data. The diagram in `docs/figures/` has its own [provenance and reuse statement](docs/figures/PROVENANCE.md).

The repository [AGPL-3.0-only license](LICENSE) applies to the SpineLocus software and does **not** override external rights in third-party materials.
