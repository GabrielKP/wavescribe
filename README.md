<h1 align="center">wavescribe</h1>

<p align="center">Program to manually correct audiofile word timings & transcriptions.</p>

<p align="center">
<a href="https://www.python.org/"><img alt="code" src="https://img.shields.io/badge/code-Python%203.12-blue?logo=Python"></a>
<a href="https://docs.astral.sh/ruff/"><img alt="Code style: Ruff" src="https://img.shields.io/badge/code%20style-Ruff-green?logo=Ruff"></a>
</p>

---

## Setup

1. Clone repository

```sh
git clone git@github.com:GabrielKP/wavescribe.git
cd wavescribe
```

2. Download audiofiles into `data/audio` and ratings into `data/ratings`.

3. Install dependencies and start program

```sh
uv install
uv run main.py
```

Outputs will be save to `data/outputs`.
