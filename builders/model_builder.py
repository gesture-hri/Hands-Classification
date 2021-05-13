from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import logging

class ModelBuilder:
    def __init__(self, config):
        self.model_names = config['models']['names']
        self.model_paths = config['models']['paths']

        self.data_paths = config['dataset']['paths']

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("model builder")

    def build_models(self):
        loaded = [np.load(path) for path in self.data_paths]

        X = np.concatenate(loaded, axis=0)
        y = np.concatenate([np.full(len(data), i) for i, data in enumerate(loaded)], axis=0)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        models = list(map(lambda x: getattr(self, x)(X_train, X_test, y_train, y_test), self.model_names))

        for model, model_path in zip(models, self.model_paths):
            pickle.dump(model, open(model_path, 'w+b'))



    