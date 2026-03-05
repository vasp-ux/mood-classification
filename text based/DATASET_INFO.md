# GoEmotions Ukrainian Dataset

Ukrainian translation of the GoEmotions dataset for emotion classification in text.

## Dataset Description

This dataset is a high-quality Ukrainian translation of Google's [GoEmotions dataset](https://github.com/google-research/google-research/tree/master/goemotions), which contains Reddit comments labeled with 28 emotion categories.

### Translation Methodology

- **Model**: [Helsinki-NLP/opus-mt-en-uk](https://huggingface.co/Helsinki-NLP/opus-mt-en-uk) - specialized English-Ukrainian translation model
- **Post-processing**: Manual quality tuning and refinement to ensure natural Ukrainian phrasing
- **Quality**: 100% Ukrainian text with natural, context-aware translations

### Dataset Statistics

- **Total samples**: 54,263 Reddit comments
- **Language**: Ukrainian (translated from English)
- **Emotion categories**: 28 + neutral
- **Splits**: Train (43,410), Validation (5,426), Test (5,427)
- **Task type**: Multi-label classification (texts can have multiple emotions)

## Emotion Categories

The dataset includes 28 emotion categories:

| Category      | Ukrainian                  | Category       | Ukrainian                  |
| ------------- | -------------------------- | -------------- | -------------------------- |
| admiration    | захоплення       | amusement      | розвага             |
| anger         | гнів                   | annoyance      | роздратування |
| approval      | схвалення         | caring         | турбота             |
| confusion     | розгубленість | curiosity      | цікавість         |
| desire        | бажання             | disappointment | розчарування   |
| disapproval   | несхвалення     | disgust        | відраза             |
| embarrassment | збентеження     | excitement     | збудження         |
| fear          | страх                 | gratitude      | вдячність         |
| grief         | горе                   | joy            | радість             |
| love          | любов                 | nervousness    | нервозність     |
| optimism      | оптимізм           | pride          | гордість           |
| realization   | усвідомлення   | relief         | полегшення       |
| remorse       | каяття               | sadness        | сум                     |
| surprise      | здивування       | neutral        | нейтрально       |

## File Structure

### CSV Format

The dataset is provided in CSV format with the following columns:

```csv
text,text_uk,labels,id,split
```

- **text**: Original English text
- **text_uk**: Ukrainian translation
- **labels**: List of emotion label indices (0-27, multi-label)
- **id**: Unique identifier
- **split**: Data split (train/validation/test)

### Example

```csv
text,text_uk,labels,id,split
"My favourite food is anything I didn't have to cook myself.","Моя улюблена їжа - це все, що я не мусив сам готувати.",[27],eebbqej,train
```

## Usage

### Loading the Dataset

```python
import pandas as pd

# Load dataset
df = pd.read_csv('goemotions_uk.csv')

# Parse labels
import ast
df['labels'] = df['labels'].apply(ast.literal_eval)

# Split by data split
train_df = df[df['split'] == 'train']
val_df = df[df['split'] == 'validation']
test_df = df[df['split'] == 'test']
```

### Multi-label Classification

```python
from sklearn.preprocessing import MultiLabelBinarizer

# Convert labels to multi-hot encoding
mlb = MultiLabelBinarizer(classes=list(range(28)))
mlb.fit([list(range(28))])

train_labels = mlb.transform(train_df['labels'])
val_labels = mlb.transform(val_df['labels'])
test_labels = mlb.transform(test_df['labels'])
```

### With Transformers

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Use multilingual models
model_name = "xlm-roberta-base"  # or "TurkuNLP/bert-base-ukrainian-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=28,
    problem_type="multi_label_classification"
)

# Tokenize Ukrainian text
encodings = tokenizer(
    train_df['text_uk'].tolist(),
    truncation=True,
    padding=True,
    max_length=128
)
```

## Applications

This dataset is suitable for:

- **Emotion detection** in Ukrainian social media and text
- **Sentiment analysis** with fine-grained emotional categories
- **Multi-label text classification** research
- **Ukrainian NLP** model development and evaluation
- **Cross-lingual emotion recognition** studies

## Citation

If you use this dataset, please cite the original GoEmotions paper:

```bibtex
@inproceedings{demszky2020goemotions,
  title={{GoEmotions: A Dataset of Fine-Grained Emotions}},
  author={Demszky, Dorottya and Movshovitz-Attias, Dana and Ko, Jeongwoo and Cowen, Alan and Nemade, Gaurav and Ravi, Sujith},
  booktitle={58th Annual Meeting of the Association for Computational Linguistics (ACL)},
  year={2020}
}
```

## License

This dataset is released under **CC0 1.0 Universal (Public Domain Dedication)**, the same license as the original GoEmotions dataset.

You are free to:

- ✅ Use commercially
- ✅ Modify and create derivatives
- ✅ Distribute
- ✅ Use privately

**No attribution required** (though appreciated in academic work).

## Acknowledgments

- Google Research for the original GoEmotions dataset
- Helsinki-NLP for the opus-mt-en-uk translation model
- The Hugging Face community for NLP tools and infrastructure

## Version

**Current Version**: 2.0 (Helsinki-NLP Edition)

- High-quality Ukrainian translations using specialized translation model
- Manual quality refinement for natural phrasing
- 100% Ukrainian text coverage

---

For questions, issues, or contributions, please open an issue on the dataset repository.
