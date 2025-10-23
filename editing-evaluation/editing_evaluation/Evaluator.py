from abc import ABC, abstractmethod

class Evaluator(ABC):
    def __init__(self, *args, **kwargs):
        if 'model_name' not in kwargs:
            raise ValueError('Editor must have a name')
        self.name = kwargs['model_name']
        self.model = None

    @abstractmethod
    def find_differences(self, img_original, img_edited, *args, **kwargs):
        pass

    @abstractmethod
    def evaluate_image(self, img_original, img_edited, edit_prompt, *args, **kwargs):
        pass

