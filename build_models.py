from builders.model_builder import ModelBuilder
from decorators.model_builder_decorators import ModelBuilderDecorators

import json

if __name__ == '__main__':
    config = json.load(open('config.json'))

    model_builder = ModelBuilder(config)
    model_builder.build_models()