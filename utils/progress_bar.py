class Dummy_Bar:
  def __init__(self, iterable=None):
    self.it = iterable

  def __call__(self, iterable):
    return iterable
  
  def __iter__(self):
    return self.it
