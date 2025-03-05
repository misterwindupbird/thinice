import logging
from typing import Callable


class AnimationManager(object):

    def __init__(self, on_finished: Callable[[], None]):
        self._blocking_animations = 0  # Use a private variable for storage
        self.on_finished = on_finished

    @property
    def blocking_animations(self):
        return self._blocking_animations

    @blocking_animations.setter
    def blocking_animations(self, value):
        assert self._blocking_animations != value
        assert value >= 0

        logging.info(f"blocking_animations changed: {self._blocking_animations} -> {value}")
        self._blocking_animations = value

        if self._blocking_animations == 0:
            self.on_finished()