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


## data schema

|  id |        text       |
|  -- | ----------------  |
|  1  | croissant cafe    |
|  2  | hotel al-naseem   |
|  3  | sami's bakery     |
|  4  | burgers and fries |

## prompt
```bash
You are a data annotator. Given a dataset, you need to annotate the dataset with the following labels:

yes: the text is about a restaurant
no: the text is not about a restaurant

Please annotate the dataset below:
```

## output schema

|  id |        text       |   label   |
|  -- | ----------------  | --------- |
|  1  | croissant cafe    |  yes      |
|  2  | hotel al-naseem   |  no       |
|  3  | sami's bakery     |  yes      |
|  4  | burgers and fries |  no       |




## to do

- [ ] add human_label to the input schema
- [ ] add a validation script using the human_label script to auto-calculate an accuracy score

