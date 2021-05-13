from decorators.preprocessor_decorators import PreprocessorDecorators
from decorators.preprocessor_decorators import Preprocessor

import pickle

class PreprocessorBuilder:
    def __init__(self, config):
        self.preprocessor_path = config['preprocessor']['path']

    def build_preprocessor(self):
        pickle.dump(Preprocessor(), open(self.preprocessor_path, "w+b"))

    