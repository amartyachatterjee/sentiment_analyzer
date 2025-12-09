from pathlib import Path

from textblob import TextBlob
from textblob import Blobber
from textblob.sentiments import NaiveBayesAnalyzer
import nltk

class SentimentAnalyzer(object):
    def __init__(self,
                 nltk_data_path: str | Path | None = None,
                 vader_threshold: float=0.05,
                 textblob_threshold: float=0.05,
                 blobber_positive_threshold: float=0.6,
                 blobber_negative_threshold: float=0.5):
        self.vader_threshold = vader_threshold
        self.textblob_threshold = textblob_threshold
        self.blobber_positive_threshold = blobber_positive_threshold
        self.blobber_negative_threshold = blobber_negative_threshold
        self.nltk_data_path = nltk_data_path

        if self.nltk_data_path is not None:
            if isinstance(self.nltk_data_path, str):
                self.nltk_data_path = Path(self.nltk_data_path)
            nltk.data.path.append(self.nltk_data_path)
        from nltk.sentiment import SentimentIntensityAnalyzer

        self.sia = SentimentIntensityAnalyzer()
        self.blobber = Blobber(analyzer=NaiveBayesAnalyzer())

    def get_vader_sentiment(self, row,text_col):
        text = row[text_col].strip()

        if text:
            result = self.sia.polarity_scores(text)
            result['vader_negative'] = result.pop('neg')
            result['vader_neutral'] = result.pop('neu')
            result['vader_positive'] = result.pop('pos')
            result['vader_overall'] = result.pop('compound')

            result['vader_overall_sentiment'] = self.get_vader_overall_sentiment_text(result['vader_overall'])

            return result
        else:
            return {
                'vader_negative': 0,
                'vader_neutral': 0,
                'vader_overall': 0,
                'vader_overall_sentiment': 'na',
                'vader_positive': 0
            }

    def get_vader_overall_sentiment_text(self, vader_overall):
        if vader_overall > self.vader_threshold:
            vader_overall_sentiment = 'positive'
        elif vader_overall < 0:
            vader_overall_sentiment = 'negative'
        else:
            vader_overall_sentiment = 'neutral'

        return vader_overall_sentiment

    def get_textblob_sentiment(self, row,text_col):
        text = row[text_col].strip()

        if text:
            blob = TextBlob(text)
            result = {'textblob_polarity': blob.sentiment.polarity,
                      'textblob_subjectivity': blob.sentiment.subjectivity}

            result['textblob_overall_sentiment'] = self.get_textblob_overall_sentiment_text(result['textblob_polarity'])

            return result
        else:
            return {
                'textblob_overall_sentiment': 'na',
                'textblob_polarity': 0,
                'textblob_subjectivity': 0
            }

    def get_textblob_overall_sentiment_text(self, textblob_polarity):
        if textblob_polarity > self.textblob_threshold:
            textblob_overall_sentiment = 'positive'
        elif textblob_polarity < 0:
            textblob_overall_sentiment = 'negative'
        else:
            textblob_overall_sentiment = 'neutral'

        return textblob_overall_sentiment

    def get_blobber_sentiment(self, row,text_col):
        text = row[text_col].strip()

        if text:
            blob = self.blobber(text)
            result = {'blobber_classification': blob.sentiment.classification,
                      'blobber_p_pos': blob.sentiment.p_pos,
                      'blobber_p_neg': blob.sentiment.n_neg}

            result['blobber_overall_sentiment'] = self.get_blobber_overall_sentiment_text(result)

            return result
        else:
            return {
                'blobber_classification': 'na',
                'blobber_overall_sentiment': 'na',
                'blobber_p_neg': 0,
                'blobber_p_pos': 0
            }

    def get_blobber_overall_sentiment_text(self, blob):
        if not blob['blobber_classification'] and blob['blobber_p_neg'] > blob['blobber_p_pos']:
            blobber_overall_sentiment = 'negative'
        elif blob['blobber_classification'] and blob['blobber_classification'] == 'neg':
            blobber_overall_sentiment = 'negative'
        elif blob['blobber_p_pos'] > self.blobber_positive_threshold:
            blobber_overall_sentiment = 'positive'
        else:
            blobber_overall_sentiment = 'neutral'

        return blobber_overall_sentiment

    def get_ensemble_sentiment(self, row):
        if (row['vader_overall_sentiment'] == 'negative' or
            row['textblob_overall_sentiment'] == 'negative' or
            row['blobber_overall_sentiment'] == 'negative'):
            ensemble_sentiment = 'negative'
        elif (row['vader_overall_sentiment'] == 'na' or
              row['textblob_overall_sentiment'] == 'na' or
              row['blobber_overall_sentiment'] == 'na'):
            ensemble_sentiment = 'na'
        elif (row['vader_overall_sentiment'] == 'neutral' or
              row['textblob_overall_sentiment'] == 'neutral' or
              row['blobber_overall_sentiment'] == 'neutral'):
            ensemble_sentiment = 'neutral'
        else:
            ensemble_sentiment = 'positive'

        return ensemble_sentiment

    def get_dummy_columns(self,df):
        column_dict = {
            'vader_negative': 0,
            'vader_neutral': 0,
            'vader_overall': 0,
            'vader_overall_sentiment': 'na',
            'vader_positive': 0,
            'textblob_overall_sentiment': 'na',
            'textblob_polarity': 0,
            'textblob_subjectivity': 0,
            'blobber_classification': 'na',
            'blobber_overall_sentiment': 'na',
            'blobber_p_neg': 0,
            'blobber_p_pos': 0,
            'ensemble_overall_sentiment': 'na'
        }

        for column_name, default_value in column_dict.items():
            df[column_name] = default_value

        return df