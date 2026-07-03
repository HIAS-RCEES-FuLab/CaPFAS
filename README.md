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

We release a complete end-to-end analytical platform for PFAS non-target screening, spanning the entire workflow from raw mass spectrometry data to final result interpretation. The package includes full source code, data processing pipelines, and supporting datasets for large-scale PFAS structural and spectral analysis.

<img width="2522" height="2388" alt="workflow" src="https://github.com/user-attachments/assets/e6b0e6e7-ec69-4a1c-9966-fbace88d7cc6" />

<img width="757" height="667" alt="platform" src="https://github.com/user-attachments/assets/6fc0bbeb-e83f-47a5-8585-db0e00c52f26" />
