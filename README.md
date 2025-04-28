# llm-data-annotation
Using llms to automate the annotation/labeling/classification of datasets given a prompt and classification scheme.


## Installation (uses conda)

```bash
conda env create -f environment.yml
conda env update -f environment.yml --prune

```

## run code
```bash
python run.py --data {}

python -m src.main --data data/sample_data.csv```

```