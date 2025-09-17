# -*- coding: utf-8 -*-
class EMA:
    def __init__(self, a=0.8):
        self.a = a
        self.v = None

    def update(self, x):
        self.v = x if self.v is None else (self.a * self.v + (1 - self.a) * x)

    def get(self):
        return 0.0 if self.v is None else float(self.v)
