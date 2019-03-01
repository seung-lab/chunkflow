

class OperatorBase(object):
    """Real Operator should inherit from this base class."""
    def __init__(self, name=''):
        assert isinstance(name, str)
        self.name=name

    def __call__(self, chunk):
        """The processing should happen in this function."""
        pass
