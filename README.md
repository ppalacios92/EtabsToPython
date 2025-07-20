# EtabsToPython

A Python-based tool for extracting geometry, section properties, and structural analysis results directly from ETABS models using the COM API. This package is designed for structural engineers and researchers who aim to automate the post-processing, visualization, and interpretation of ETABS models using Python.

## Features

- Connects directly to an open ETABS session.
- Extracts:
  - Story definitions and point connectivity
  - Beam, column, brace, wall, and floor object connectivity
  - Frame section definitions and assignments
  - Modal participating mass ratios
  - Element and story forces
  - Joint displacements
- Builds a full geometric and analytical representation using `pandas` DataFrames.
- Assigns section-based colors and dimensions for 3D visualization and reconstruction.

## Requirements

- ETABS installed and running (compatible with COM API)
- Python 3.8+
- Required Python packages:
  - `comtypes`
  - `numpy`
  - `pandas`
  - `matplotlib` (for optional 3D visualization)

## Installation

Clone the repository:

```bash
git clone https://github.com/ppalacios92/EtabsToPython.git
cd EtabsToPython
