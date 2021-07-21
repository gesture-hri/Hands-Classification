from typing import Tuple, List, Iterator
from itertools import groupby, filterfalse, starmap
from multiprocessing import Queue

import numpy as np
import logging
import pickle

from time import time


class DatasetBuilder:
    def __init__(self, config):
        self.labels = config['dataset']['labels']
        self.label_paths = config['dataset']['paths']
        self.preprocessor = pickle.load(open(config['preprocessor']['path'], 'rb'))

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("dataset preparer")

    @staticmethod
    def preprocessor_worker(label, frames, config, q: Queue):
        preprocessor = pickle.load(open(config['preprocessor']['path'], 'rb'))
        processed_frames = list(filterfalse(lambda x: x is None, map(lambda x: preprocessor.preprocess(x), frames)))
        q.put((label, processed_frames))
        
    def separate_frames_by_label(self, raw_dataset_iterable: Iterator[Tuple[np.ndarray, int]]):
        separated_labels = [
            list(map(lambda x: x[0], frames)) for _label, frames in groupby(
                sorted(raw_dataset_iterable, key=lambda x: x[1]), key=lambda x: x[1]
            )
        ]
        return separated_labels

    def prepare_dataset(self, raw_dataset_iterable: Iterator[Tuple[np.ndarray, int]]):
        '''
        Keep this method for backwards compatibility
        '''
        separated_frames = self.separate_frames_by_label(raw_dataset_iterable)
        preprocessed_frames = [list(filterfalse(lambda x: x is None, 
                                    map(lambda x: self.preprocessor.preprocess(x), frames))
                                ) for frames in separated_frames]
        
        self.logger.info("Succesfully preprocessed {} frames in total".format(
            sum([len(label_frames) for label_frames in preprocessed_frames])
        ))
        
        for label_path, label_frames in zip(self.label_paths, preprocessed_frames):
            np.save(label_path, label_frames)



        