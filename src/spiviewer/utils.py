import time


class Gear:
    def __init__(self, speed, deltaTimes):
        self._lastTick = time.time()
        self.speed = speed
        self.deltaTimes = deltaTimes

    def getSpeed(self):
        t = time.time()
        self._lastTick, dt = t, t - self._lastTick
        for s, d in zip(self.speed, self.deltaTimes):
            if dt < d:
                return s
        return self.speed[-1]
