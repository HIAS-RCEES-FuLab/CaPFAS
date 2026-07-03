# CaPFAS

CaPFAS (Comprehensive Analysis of PFAS) is an integrated platform for comprehensive PFAS screening aligned with the OECD definition, comprising an interpretable multimodal neural network and a downstream structural annotation module.

This repository provides an implementation of the platform, including data cleaning, preprocessing, model training, screening methods, and result visualization.

<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/2a8cdecb-0de6-4a09-a813-b9561868eba8" />

---

## Spectral Libraries

The spectral databases used in this project include:

- **NIST 2020 MS/MS Library**
  - The library can be exported by following the MassFormer instructions:
  - https://github.com/Roestlab/massformer?tab=readme-ov-file#exporting-the-nist-data

- **GNPS Spectral Libraries**
  - Available at:
  - https://external.gnps2.org/gnpslibrary

- **MoNA (MassBank of North America)**
  - Available at:
  - https://mona.fiehnlab.ucdavis.edu/downloads

---

## Platform

## Platform

The graphical user interface of CaPFAS is shown below.

<img width="757" height="667" alt="platform" src="https://github.com/user-attachments/assets/6fc0bbeb-e83f-47a5-8585-db0e00c52f26" />

The platform consists of three main modules: **Peak Feature Mining**, **Feature Filtering**, and **Identification**.

### Peak Feature Mining Parameter Settings

This module is used for MS feature extraction and preprocessing.

- **Scan mode**
  - **DDA** is the recommended acquisition mode because each MS/MS spectrum is directly associated with its precursor ion.
  - **DIA** and **Full Scan** data are also supported. For DIA data, CaPFAS performs model prediction using the acquired window-based MS/MS spectra because precursor-specific MS/MS spectra are unavailable. For Full Scan data, where no MS/MS spectra are acquired, the model directly uses the acquired MS spectra for prediction.

- **Ion mode**
  - Select the ionization mode according to the experimental data.
  - **The current PFAS identification model is trained for negative ion mode only**, and therefore negative mode is recommended for PFAS analysis.

- **Noise threshold**
  - Defines the intensity threshold for noise removal.
  - This parameter should be adjusted according to the performance and noise characteristics of the mass spectrometer used.

- **Reverse analysis**
  - Directly analyzes all acquired MS/MS spectra without performing peak feature extraction.

- **Extract MS/MS**
  - Extracts MS/MS spectra associated with detected features for downstream analysis.

- **Single-trace filtering**
  - Uses the OpenMS single-trace filtering algorithm to remove low-confidence features and reduce potential false-positive peaks.

- **Adduct annotation**
  - Annotates supported adduct ions according to predefined adduct rules.

### Feature Filtering

This module performs candidate screening.

- **Filtering method**
  - **PFAS ML** (default): the multimodal CaPFAS model developed in this work.
  - Traditional screening methods are also available, including:
    - Mass defect filtering
    - Diagnostic fragment ion filtering
    - Neutral loss filtering
  - These methods can be combined using the options in **Unit Settings**.

### Identification

This module performs hierarchical compound identification.

1. **Exact mass and isotope pattern matching** (Level 1 candidate screening).
2. **Theoretical fragment prediction and matching**.
3. **MS/MS spectral matching** against reference spectra.

All matching parameters can be customized according to the analytical requirements.

---

## Release

The GitHub Release provides a complete end-to-end analytical platform for PFAS non-target screening, spanning the entire workflow from raw mass spectrometry data to final result interpretation.

The release includes:

- Complete source code
- Executable programs
- Data processing workflows
- Supporting datasets
- Large-scale PFAS structural and spectral databases
- Support for integrating user-provided datasets

After downloading and extracting the Release package, simply launch the executable program to start the CaPFAS platform. The graphical user interface shown above provides access to the complete workflow, from raw mass spectrometry data processing to final PFAS screening and structural annotation.

<img width="757" height="667" alt="workflow" src="https://github.com/user-attachments/assets/e6b0e6e7-ec69-4a1c-9966-fbace88d7cc6" />
