from builders.dataset_builder import DatasetBuilder
from decorators.class_decorators import ClassDecorators
from typing import Tuple, Iterator

import numpy as np

class DatasetBuilderDecorators:

    @ClassDecorators.add_to_class(DatasetBuilder)
    def _load_tensorflow_dataset(self, **kargs) -> Iterator[Tuple[np.ndarray, int]]:
        import tensorflow_datasets as tsdf
        dataset = tsdf.load('rock_paper_scissors')

        for data in dataset['train']:
            yield data['image'].numpy(), data['label'].numpy()

    @ClassDecorators.add_to_class(DatasetBuilder)
    def load_dataset(self, **kargs) -> Iterator[Tuple[np.ndarray, int]]:
        '''
        Actual dataset loading. Overwritte this method with any self defined methods added above
        :param no specified method parameter. Loading dataset should be generic
        :return iterator to tuples of frame/frame sequence and corresponding single label 
        (also in case of frame sequence!)
        '''
        return self._load_tensorflow_dataset()
