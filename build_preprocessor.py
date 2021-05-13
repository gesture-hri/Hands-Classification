from builders.preprocessor_builder import PreprocessorBuilder

import json

if __name__ == '__main__':
    config = json.load(open('config.json'))

    builder = PreprocessorBuilder(config)
    builder.build_preprocessor()