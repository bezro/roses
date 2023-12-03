import fire

from roses.train import train as train_wrapper


def train():
    train_wrapper()


if __name__ == "__main__":
    fire.Fire()
