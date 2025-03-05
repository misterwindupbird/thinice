import logging

class AnimationManager(object):

    def __init__(self):
        self._blocking_animations = 0  # Use a private variable for storage

    @property
    def blocking_animations(self):
        return self._blocking_animations

    @blocking_animations.setter
    def blocking_animations(self, value):
        logging.info(f"blocking_animations changed: {self._blocking_animations} -> {value}")
        self._blocking_animations = value