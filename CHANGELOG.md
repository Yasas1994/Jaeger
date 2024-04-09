## version 1.1.25
---
- Added CHANGELOG.md
- pypi version 1.1.251 - reason : patch
- Better error messages and handling
- Easy installation via bioconda and pip
- Significant speed improvements compared to 1.1.23
- New prediction reliability estimation module
- New prophage prediction mode (unstable)
- New output fields 
    1. entropy (Another measure of prediction confidence; lower the better)
    2. realiability_score (confidence in the predicion ; higher the better)
    3. host_contam (whether there are host like regions)
    4. prophage_contam (whether there are phage like regions)
    5. Bacteria_var (variance of Bacteria_score)
    6. Phage_var (variance of Phage_score)
    7. Eukarya_var (variance of Eukarya_score)
    8. Archaea_var (variance of Archaea_score)

## version 1.1.23
---
- Fixed a bug that prevented GPU utilization
- Added support for manually setting the fragment size.
- Fixed wrong version number

## version 1.1.22
---
- This release bring a python API to Jaeger enabling easy integration of Jaeger to python scripts and packages.