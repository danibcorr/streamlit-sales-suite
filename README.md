# Streamline Sales Suite

A powerful, all-in-one platform for data-driven sales analysis and visualization, built
with efficiency and automation in mind.

## Overview

**Streamline Sales Suite** is a comprehensive application designed to simplify the
exploration and interpretation of sales data. Whether you're tracking KPIs, identifying
trends, or preparing reports, this tool offers a robust and interactive environment for
data analysis using [Streamlit](https://streamlit.io).

## Installation & Setup

Follow these steps to get up and running quickly:

### 1. Clone the Repository

```bash
git clone https://github.com/danibcorr/streamlit-sales-suite
cd streamlit-sales-suite
```

### 2. Set Up Your Python Environment

Create and activate a virtual environment (e.g., with `venv` or `conda`):

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
```

### 3. Install Dependencies

Use the `Makefile` for automated setup:

```bash
make
```

Or install dependencies manually:

```bash
pip install -r pyproject.toml
```

## Development & CI Pipeline

Run the full suite of quality checks:

```bash
make pipeline
```

Trigger a complete workflow, including setup, cleanup, and tests:

```bash
make all
```

## Feedback & Contributions

We welcome issues, suggestions, and pull requests! Please open an
[issue](https://github.com/danibcorr/streamlit-sales-suite/issues) if you encounter any
bugs or have improvement ideas.
