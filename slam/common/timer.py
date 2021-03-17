import torch
import time


class Duration:
    def __init__(self, title: str = ""):
        self.duration_ms: int = 0
        self.title = title
        self.iters = 0

    def get_message(self) -> str:
        return f"""
-----------------------------------------------
ELAPSED {self.title} : 
-----------------------------------------------
milliseconds {self.duration_ms}
seconds      {self.duration_ms / 1000}
minutes      {self.duration_ms / (1000 * 60)}
hours        {self.duration_ms / (1000 * 3600)} 
----------------------------------------------- 
Num Iters :  {self.iters}
s/it         {self.duration_ms / (1000 * max(1, self.iters))}
----------------------------------------------- 
        """

    def __del__(self):
        print(self.get_message())


def timer() -> object:
    def __decorator(func):
        if not hasattr(func, "duration"):
            duration = Duration(func.__name__)
            setattr(func, "duration", duration)
        else:
            duration = getattr(func, "duration")

        def __wrapper(*args, **kwargs):
            duration.iters += 1
            start = time.clock()

            result = func(*args, **kwargs)

            elapsed = time.clock() - start
            duration.duration_ms += elapsed * 1000

            return result

        return __wrapper

    return __decorator


def torch_timer(title: str = "", cuda: bool = True):
    duration = Duration(title)

    def decorator(func):
        def wrapper(*args, **kwargs):
            duration.iters += 1
            if cuda:
                torch.cuda.synchronize()

            start = time.clock()

            result = func(*args, **kwargs)
            if cuda:
                torch.cuda.synchronize()

            elapsed = time.clock() - start
            duration.duration_ms += elapsed * 1000

            return result

        return wrapper

    return decorator
