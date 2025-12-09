# sentiment_analyzer
Performs basic sentiment analysis (Vader, TextBlob, Blobber and an ensemble) of input text

## **Installation**
```
pip install git+https://github.com/amartyachatterjee/sentiment_analyzer.git
```

## **Parameters**
```
- nltk_data_path: str | Path | None = None -> Refer to below example
- vader_threshold: float = 0.05 -> Positive if Vader compound > vader_threshold
- textblob_threshold: float = 0.05 -> Positive if TextBlob polarity > textblob_threshold
- blobber_positive_threshold: float = 0.6 -> Positive if Blobber (Naive Bayes) p_pos > blobber_positive_threshold and p_pos > p_neg
- blobber_negative_threshold: float = 0.5 -> Currently not used
```
### **Returns**
```
**For Vader:**
{
  'vader_negative': float,
  'vader_neutral': float,
  'vader_overall': float,
  'vader_overall_sentiment': str, # positive/negative/neutral
  'vader_positive': float
}

**For TextBlob:**
{
  'textblob_overall_sentiment': str, # positive/negative/neutral
  'textblob_polarity': float,
  'textblob_subjectivity': float
}

**For Blobber:**
{
  'blobber_classification': str,
  'blobber_overall_sentiment': str, # positive/negative/neutral
  'blobber_p_neg': float,
  'blobber_p_pos': float
}

**For ensemble:**
ensemble_sentiment: str # positive/negative/neutral
```

## **Usage**
```
from pathlib import Path
import pandas as pd
from sentiment_analyzer.sentiment_analyzer.sentiment_analyzer import SentimentAnalyzer

def run_sentiment_analyser(df,
                           sentiment_analyzer):
  print(f'''\t**** Running sentiment analyser now, starting with Vader sentiment ****''')

  if len(df) <= 0:
      df = sentiment_analyzer.get_dummy_columns(df)
      print(f'''\t\tNo data available to run sentiment analysis.''')
      return df

  tmp_df = df.apply(lambda row: self.sentiment_analyzer.get_vader_sentiment(row,
                                                                            text_col='scrubbed_text'), 
                    axis=1, 
                    result_type='expand')
  df = pd.concat([df,tmp_df], axis = 1)
  print(f'''\t\tCollected Vader sentiment. Next collecting TextBlob sentiment.''')
  tmp_df = df.apply(lambda row: self.sentiment_analyzer.get_textblob_sentiment(row,
                                                                               text_col='scrubbed_text'), 
                    axis=1, 
                    result_type='expand')
  df = pd.concat([df,tmp_df], axis = 1)
  print(f'''\t\tCollected TextBlob sentiment. Next collecting Blobber (Naive Bayes) sentiment.''')
  tmp_df = df.apply(lambda row: self.sentiment_analyzer.get_blobber_sentiment(row,
                                                                              text_col='scrubbed_text'), 
                    axis=1, 
                    result_type='expand')
  df = pd.concat([df,tmp_df], axis = 1)
  del tmp_df
  gc.collect()
  print(f'''\t\tCollected Blobber (Naive Bayes) sentiment. Next collecting ensemble sentiment.''')
  df['ensemble_overall_sentiment'] = df.apply(lambda row: self.sentiment_analyzer.get_ensemble_sentiment(row), 
                                              axis=1)
  print(f'''\t\tCollected ensemble sentiment.''')

  return df

# Mainline
nltk_data_path = Path("C:/path/to/nltk_data")
sentiment_analyzer = SentimentAnalyzer(nltk_data_path=self.nltk_data_path)

df = run_sentiment_analyzer(pd.read_excel(Path('C:/path/to/data/file')),
                            sentiment_analyzer)
