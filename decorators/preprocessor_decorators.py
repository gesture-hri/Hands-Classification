from decorators.class_decorators import ClassDecorators
from typing import Union
import numpy as np


class Preprocessor:
    def __init__(self):
        pass


class PreprocessorDecorators:

    @ClassDecorators.add_to_class(Preprocessor)
    def _mediapipe_process(self, frame: np.ndarray) -> Union[np.ndarray, None]:
        import mediapipe
        with mediapipe.solutions.hands.Hands(static_image_mode=True, max_num_hands=1) as hands_estimator:
            processed = hands_estimator.process(frame)
            if not processed or not processed.multi_hand_landmarks:
                return None

            hand_coordinates = [
                [landmark.x, landmark.y, landmark.z] 
                for hand in processed.multi_hand_landmarks
                for landmark in hand.landmark
            ]

            return hand_coordinates

    @ClassDecorators.add_to_class(Preprocessor)
    def _flatten_frame(self, processed: np.ndarray) -> Union[np.ndarray, None]:
        from itertools import chain

        if processed is None:
            return None
        return np.array(list(chain.from_iterable(
            processed
        )))

    @ClassDecorators.add_to_class(Preprocessor)
    def preprocess(self, frame: np.ndarray) -> Union[np.ndarray, None]:
        '''
        Actual preprocessing. Overwritte this method with any self-defined methods added above
        :param frame: image or sequence of images that can be classified
        :return numpy array from preprocessed frame/frame sequence | None if preprocessing failed
        '''
        return self._flatten_frame(self._mediapipe_process(frame))