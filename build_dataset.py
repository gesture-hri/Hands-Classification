from builders.dataset_builder import DatasetBuilder
from decorators.dataset_builder_decorators import DatasetBuilderDecorators

import json

if __name__ == '__main__':
    config = json.load(open('config.json'))
    
    dataset_builder = DatasetBuilder(config)
    dataset_builder.prepare_dataset(
        dataset_builder.load_dataset()
    )