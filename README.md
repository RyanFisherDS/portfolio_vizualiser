# portfolio_viz

A small Python repository for visualizing Morningstar-style box portfolio exposures.

## Overview

This repo contains a utility class for representing 3x3 Morningstar style boxes and blending multiple style exposures into a portfolio.

## Contents

- `stylebox.py` — Defines `StyleBox` class and a `blend()` function for creating blended portfolios.

## Requirements

- Python 3.8+
- `numpy`

## Setup

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install numpy
```

## Usage

Run the example in `portfolio_viz.py` directly:

```powershell
python portfolio_viz.py
```

This will display the blended style box for an example portfolio.

## Notes

- The `StyleBox` class normalises the 3x3 weight matrix if the weights do not sum to `1.0`.
- The `blend()` function automatically normalises portfolio weights.
- The example uses approximate Morningstar style allocations for illustration only.
