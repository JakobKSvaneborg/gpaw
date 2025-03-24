from gpaw.new.environment import Environment
from gpaw.new.solvation import Solvation


class SJM:
    def __init__(self,
                 cavity,
                 workfunction,
                 charge):
        ...

    def todict(self):

    def build(self, ...):
        return SJMEnvironment()


class SJMEnvironment(Environment):
    def __init__(self, solvation):
        self.solvation = solvation

