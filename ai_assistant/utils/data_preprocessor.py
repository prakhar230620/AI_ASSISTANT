# data_preprocessor.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from typing import Union, List, Dict, Any

class DataPreprocessor:
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}

    def clean_text(self, text: str) -> str:
        text = re.sub(r'[^\w\s]', '', text.lower())
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        cleaned_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
        return ' '.join(cleaned_tokens)

    def scale_features(self, data: pd.DataFrame, columns: List[str], method: str = 'standard') -> pd.DataFrame:
        for col in columns:
            if col not in self.scalers:
                self.scalers[col] = StandardScaler() if method == 'standard' else MinMaxScaler()
            data[col] = self.scalers[col].fit_transform(data[col].values.reshape(-1, 1))
        return data

    def encode_categorical(self, data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        for col in columns:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
            data[col] = self.encoders[col].fit_transform(data[col])
        return data

    def handle_missing_values(self, data: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
        for col in data.columns:
            if data[col].isnull().sum() > 0:
                if col not in self.imputers:
                    self.imputers[col] = SimpleImputer(strategy=strategy) if strategy != 'knn' else KNNImputer()
                data[col] = self.imputers[col].fit_transform(data[col].values.reshape(-1, 1))
        return data

    def reduce_dimensions(self, data: pd.DataFrame, n_components: int) -> pd.DataFrame:
        pca = PCA(n_components=n_components)
        reduced_data = pca.fit_transform(data)
        return pd.DataFrame(reduced_data, columns=[f'PC{i+1}' for i in range(n_components)])

    def select_features(self, X: pd.DataFrame, y: pd.Series, k: int) -> pd.DataFrame:
        selector = SelectKBest(score_func=f_classif, k=k)
        selected_features = selector.fit_transform(X, y)
        return pd.DataFrame(selected_features, columns=X.columns[selector.get_support()])

    def process_data(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        if config.get('text_columns'):
            for col in config['text_columns']:
                data[col] = data[col].apply(self.clean_text)

        if config.get('numeric_columns'):
            data = self.scale_features(data, config['numeric_columns'], config.get('scaling_method', 'standard'))

        if config.get('categorical_columns'):
            data = self.encode_categorical(data, config['categorical_columns'])

        if config.get('handle_missing'):
            data = self.handle_missing_values(data, config.get('imputation_strategy', 'mean'))

        if config.get('reduce_dimensions'):
            data = self.reduce_dimensions(data, config['n_components'])

        if config.get('select_features'):
            data = self.select_features(data.drop(config['target'], axis=1), data[config['target']], config['k_features'])

        return data