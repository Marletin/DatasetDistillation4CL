import contextlib
import os
from collections import namedtuple
from torchvision import datasets, transforms as T

default_dataset_roots = dict(
    MNIST="./data/mnist",
    MNIST_RGB="./data/mnist",
    SVHN="./data/svhn",
    FASHION_MNIST="./data/fashion_mnist"
)


dataset_normalization = dict(
    MNIST=((0.1307,), (0.3081,)),
    MNIST_RGB=((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)),
    FASHION_MNIST=((0.2859, 0.2859, 0.2859),(0.3530, 0.3530, 0.3530)),
    SVHN=((0.4379104971885681, 0.44398033618927, 0.4729299545288086),
          (0.19803012907505035, 0.2010156363248825, 0.19703614711761475)),
)


dataset_labels = dict(
    MNIST=list(range(10)),
    MNIST_RGB=list(range(10)),
    FASHION_MNIST=list(range(10)),
    SVHN=list(range(10)),
)

DatasetStats = namedtuple("DatasetStats", " ".join(["nc", "real_size", "num_classes"]))

dataset_stats = dict(
    MNIST=DatasetStats(1, 32, 10),
    MNIST_RGB=DatasetStats(3, 32, 10),
    FASHION_MNIST=DatasetStats(3, 32, 10),
    SVHN=DatasetStats(3, 32, 10),
)

assert(set(default_dataset_roots.keys()) == set(dataset_normalization.keys()) ==
       set(dataset_labels.keys()) == set(dataset_stats.keys()))


def get_info(dataset):
    name = dataset
    assert name in dataset_stats, f"Unsupported dataset: {dataset}"
    nc, input_size, num_classes = dataset_stats[name]
    normalization = dataset_normalization[name]
    root = default_dataset_roots[name]
    labels = dataset_labels[name]
    return name, root, nc, input_size, num_classes, normalization, labels


@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        yield


def get_dataset(phase, dataset):
    assert phase in ("train", "test"), f"Unsupported phase: {phase}"
    input_size = 32
    name, root, _, _, _, normalization, _ = get_info(dataset)

    if name == "MNIST":
        transform_list = [
            T.Resize([input_size, input_size], T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(*normalization)
            ]
        with suppress_stdout():
            return datasets.MNIST(root, train=(phase == "train"), download=True,
                                  transform=T.Compose(transform_list))

    elif name == "MNIST_RGB":
        transform_list = [
            T.Grayscale(3),
            T.Resize([input_size, input_size], T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(*normalization)
            ]
        with suppress_stdout():
            return datasets.MNIST(root, train=(phase == "train"), download=True,
                                  transform=T.Compose(transform_list))

    elif name == "FASHION_MNIST":
        transform_list = [
            T.Grayscale(3),
            T.Resize([input_size, input_size], T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(*normalization),
            ]
        with suppress_stdout():
            return datasets.FashionMNIST(root, train=(phase == "train"), download=True,
                                  transform=T.Compose(transform_list))
  
    elif name == "SVHN":
        transform_list = [
            T.ToTensor(),
            T.Normalize(*normalization),
            ]
        with suppress_stdout():
            return datasets.SVHN(root, split=phase, download=True,
                                 transform=T.Compose(transform_list))

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
