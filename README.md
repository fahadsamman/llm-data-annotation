# llm-data-annotation
Using llms to automate the annotation/labeling/classification of datasets given a prompt and classification scheme.


## Installation (uses conda)

```bash
conda env create -f environment.yml
conda env update -f environment.yml --prune

```

## run code
```bash
python run.py --data {} --batch_size {}

python -m src.main --data data/sample_data.csv``` --batch_size 10

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
You are a data annotator.

You will be given {batch_size} data points as strings.
For each string, classify it with ONLY one label: '{positive_label}' or '{negative_label}'.

Examples:
- Cafe → {positive_label}
- Restaurant → {negative_label}
- Hotel → {negative_label}
- Shisha Cafe → {negative_label}
- Bakery → {positive_label}
- Fast Food → {negative_label}

REPLY WITH EXACTLY {batch_size} LINES, ONE ANSWER PER LINE.
Only output '{positive_label}' or '{negative_label}' for each row, in the same order.
NO numbers.
NO extra text.
```

default: `batch_size=10, positive_label="yes", negative_label="no"`


## output schema

|  id |        text       |   label   |
|  -- | ----------------  | --------- |
|  1  | croissant cafe    |  yes      |
|  2  | hotel al-naseem   |  no       |
|  3  | sami's bakery     |  yes      |
|  4  | burgers and fries |  no       |
|  5  | shisha and cafe   |  no       |



## to do

- [ ] add human_label to the input schema
- [ ] add a validation script using the human_label script to auto-calculate an accuracy score

