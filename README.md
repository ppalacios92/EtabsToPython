# EtabsToPython

A Python-based tool for extracting geometry, section properties, and structural analysis results directly from ETABS models using the COM API. This package is designed for structural engineers and researchers who aim to automate the post-processing, visualization, and interpretation of ETABS models using Python.

---

## ⚙️ Features

- Connects directly to an open ETABS session via the COM API.
- Extracts the following information:
  - Story definitions and point coordinates
  - Beam, column, brace, wall, and floor object connectivity
  - Frame section definitions and assignments
  - Modal participating mass ratios
  - Element and story-level forces
  - Joint displacements and rotations
- Provides full geometric and analytical representation using pandas DataFrames.
- Enables visualization and structural interpretation using custom plotting tools.
- Supports automatic section tagging and colorization for 3D reconstruction.

---

## 📦 Requirements

- ETABS (must be installed and running)
- Windows OS with COM API enabled
- Python 3.8 or higher
- Python libraries:
  - comtypes
  - numpy
  - pandas
  - matplotlib (optional for visualization)

---

## 🚀 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/ppalacios92/EtabsToPython.git
cd EtabsToPython
pip install -e .
```
---

## 📁 Repository Structure
```bash
EtabsToPython/
├── core/                 # Core ETABS model extraction and data handling
├── tools/                # Plotting and post-processing utilities
├── examples/             # Jupyter notebooks with usage examples
├── config/               # Optional configuration files
├── tests/                # Unit tests and validation
└── README.md             # Project documentation
```

---
## 🛑 Disclaimer
This tool is provided as-is, without any guarantees of accuracy, performance, or suitability for specific engineering tasks.
The author assumes no responsibility for the interpretation of results, post-processing errors, or consequences of incorrect data extraction.
Use at your own risk and always validate against the ETABS graphical environment and design codes.

---
## 👨‍💻 Author

Developed by Patricio Palacios B.
Structural Engineer | Python Developer | Seismic Modeler
GitHub: @ppalacios92

## 📚 How to Cite

If you use this tool in your work, please cite it as follows:

```bibtex
@misc{palacios2025etabstopython,
  author       = {Patricio Palacios B.},
  title        = {EtabsToPython: A Python-based ETABS data extraction and visualization tool},
  year         = {2025},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/ppalacios92/EtabsToPython}}
}
```

## 📄 Citation in APA (7th Edition)

Palacios B., P. (2025). *EtabsToPython: A Python-based ETABS data extraction and visualization tool* [Computer software]. GitHub. https://github.com/ppalacios92/EtabsToPython

---

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests, report bugs, or suggest new features through the GitHub issues page.
