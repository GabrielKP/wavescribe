<h1 align="center">wavescribe</h1>

<p align="center">Program to manually correct audiofile word timings & transcriptions.</p>

<p align="center">
<a href="https://www.python.org/"><img alt="code" src="https://img.shields.io/badge/code-Python%203.12-blue?logo=Python"></a>
<a href="https://docs.astral.sh/ruff/"><img alt="Code style: Ruff" src="https://img.shields.io/badge/code%20style-Ruff-green?logo=Ruff"></a>
</p>

---

## Setup

1. Clone repository (you will need [git](https://git-scm.com/install/))

    - Open console & navigate to directory of choice.

```sh
git clone git@github.com:GabrielKP/wavescribe.git
cd wavescribe
```

2. Install uv OR setup a virtual environment

    - **(a)** uv is a package manager which handles dependencies and virtual environments: [Install uv](https://docs.astral.sh/uv/getting-started/installation/)
    - After you install uv it will automatically handle virtual environments for you when you run the program, nothing more is needed!

    - **OR (b)** set up a virtual environment with conda.

```sh
conda create -n wavescribe python=3.12 -y
conda activate wavescribe
# install dependencies
pip install .
```

3. Place audio files into `data/audio`

    - Audio files **must start with `sub-`** and have number following that (e.g. `sub-001` or `sub-071`)


4. Place pre-annotated timestamp files into `data/pre_annotated`

    - Pre-annotated files **must start with `sub-`** and have a number following that, but everything after the number can be custom (e.g. `sub-001-annotation.csv`, or `sub-078-transcript.csv`).
    - Each pre-annotated file is expected to be a csv file that contaisn a word or text segment in each row and the following columns:
        - **transcription**: text of the word/text segment
        - **start**: approximate start time of word/text segment
        - **end**: approximate end time of the word/text segment


5. Run program & rate

```sh
# with uv
uv run main.py

# conda/other virtualenv
python main.py
```

    - You will have to choose your data folder, usually it will already be the correct one.
    - Put your rater initials/name in the field, each word you rate will be tracked.
    - Click 'load' for the subjects which show on the left sides.
    - Annotations are saved automatically after pressing `next` or `previous`.


6. Outputs

    - Outputs will be save to `data/outputs`.
    - Output files will be named the same as the corresponding files in `data/pre_annotated`. E.g. (`sub-001` with pre_annotated file `data/pre_annotated/sub-001-annotation.csv` will have an output file `data/output/sub-001-annotation.csv` or `sub-078` with file `data/pre_annotated/sub-078-transcript.csv` will have an output file `data/output/sub-078-transcript.csv`)
    - Each output file will contain the following columns:
        - **transcription**: the transcription typed by the rater.
        - **word_clean**: the transcription in lower-case, and stripped with dots and spaces.
        - **start**: start time of the rated word.
        - **end**: end time of the rated word.
        - **rater**: the last rater for the word.
        - **changed**: whether the rater has changed the word from the previous rating.
        - Any other column which was in the pre_annoated file will also be present in the output file.



Other information:
- If you want to change your rater initials or data dir, you have to delete or edit [wavescribe_settings.json](wavescribe_settings.json).
