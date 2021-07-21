from builders.dataset_builder import DatasetBuilder
from decorators.dataset_builder_decorators import DatasetBuilderDecorators
from multiprocessing import Process
from multiprocessing import Queue

import json
import pickle
import numpy as np

from time import time

if __name__ == '__main__':
    config = json.load(open('config.json'))
    
    dataset_builder = DatasetBuilder(config)
    separated_frames = dataset_builder.separate_frames_by_label(dataset_builder.load_dataset())
    ipc_queue = Queue()

    children = [
        Process(target=DatasetBuilder.preprocessor_worker, args=(label, frames, config, ipc_queue))
        for label, frames in enumerate(separated_frames)
    ]

    for child in children:
        child.start()

    dataset = [ipc_queue.get() for _ in dataset_builder.labels]

    dataset_builder.logger.info("Sucesfully preprocessed {} frames in total.".format(
        sum(len(frames) for _label, frames in dataset)
    ))
    
    for label, frames in dataset:
        np.save(dataset_builder.label_paths[label], frames)
