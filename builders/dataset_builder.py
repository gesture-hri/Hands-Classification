from typing import Tuple, List, Iterator

import numpy as np
import logging
import pickle
import itertools

class DatasetBuilder:
    def __init__(self, config):
        self.labels = config['dataset']['labels']
        self.label_paths = config['dataset']['paths']
        self.preprocessor = pickle.load(open(config['preprocessor']['path'], 'rb'))

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("dataset preparer")

    def prepare_dataset(self, raw_dataset_iterable: Iterator[Tuple[np.ndarray, int]]):
        
        separated_labels = [list(map(lambda x: x[0], frames)) for _k, frames in itertools.groupby(
            filter(lambda x: x[0] is not None,
                sorted([(self.preprocessor.preprocess(frame), label) for frame, label in raw_dataset_iterable], 
                    key=lambda x: x[1])
            ),key=lambda x: x[1])
        ]

        self.logger.info("Succesfully preprocessed {} frames in total".format(
            sum([len(label_frames) for label_frames in separated_labels])
        ))
        
        for label_path, label_frames in zip(self.label_paths, separated_labels):
            np.save(label_path, label_frames)



        