import torch


models = {}

def register(name):
  def decorator(cls):
    models[name] = cls
    return cls
  return decorator


def make(name, **kwargs):
  if name in models:
    model = models[name](**kwargs)
  else:
    raise ValueError("Unknown model {:s}".format(name))
  if torch.cuda.is_available():
    model.cuda()
  return model
