import argparse
import logging
import os
import time
from contextlib import contextmanager

import torch
import yaml

import datasets
import utils


class State(object):
    class UniqueNamespace(argparse.Namespace):
        def __init__(self, requires_unique=True):
            self.__requires_unique = requires_unique
            self.__set_value = {}

        def requires_unique(self):
            return self.__requires_unique

        def mark_set(self, name, value):
            if self.__requires_unique and name in self.__set_value:
                raise argparse.ArgumentTypeError(
                    f"'{name}' appears several times: {self.__set_value[name]}, {value}.")
            self.__set_value[name] = value

    __inited = False

    def __init__(self, opt=None):
        if opt is None:
            self.opt = State.UniqueNamespace()
        else:
            if isinstance(opt, argparse.Namespace):
                opt = vars(opt)
            self.opt = argparse.Namespace(**opt)
        self.extras = {}
        self.__inited = True

    def __setattr__(self, k, v):
        if not self.__inited:
            return super(State, self).__setattr__(k, v)
        else:
            self.extras[k] = v

    def __getattr__(self, k):
        if k in self.extras:
            return self.extras[k]
        elif k in self.opt:
            return getattr(self.opt, k)
        raise AttributeError(k)

    def copy(self):
        return argparse.Namespace(**self.merge())

    @contextmanager
    def pretend(self, **kwargs):
        saved = {}
        for key, val in kwargs.items():
            if key in self.extras:
                saved[key] = self.extras[key]
            setattr(self, key, val)
        yield
        for key, val in kwargs.items():
            self.pop(key)
            if key in saved:
                self.extras[key] = saved[key]

    def pop(self, k, default=None):
        return self.extras.pop(k, default)

    def clear(self):
        self.extras.clear()

    # returns a single dict containing both opt and extras
    def merge(self, public_only=False):
        vs = vars(self.opt).copy()
        vs.update(self.extras)
        if public_only:
            for k in tuple(vs.keys()):
                if k.startswith("_"):
                    vs.pop(k)
        return vs

    def get_base_directory(self, mode=None, dataset=None):
        vs = self.merge()
        opt = argparse.Namespace(**vs)
        name = ""
        if opt.mode == "distill_adapt" and mode is None:
            name = f"Source_{opt.source_dataset}"
        dirs = [opt.mode, opt.dataset, name]
        if mode is not None and dataset is not None:
            dirs = [mode, dataset, name]
        return os.path.join("./results/", *dirs)

    def get_load_directory(self, mode=None, dataset=None):
        return self.get_base_directory(mode, dataset)

    def get_save_directory(self, mode=None, dataset=None):
        base_dir = self.get_base_directory(mode, dataset)
        if self.phase == "test":
            base_dir = os.path.join(base_dir, "test")
        return base_dir

    def get_model_dir(self):
        vs = vars(self.opt).copy()
        vs.update(self.extras)
        opt = argparse.Namespace(**vs)
        model_dir = "./models/"
        if opt.mode == "distill_adapt":
            dataset = opt.source_dataset
        else:
            dataset = opt.dataset
        subdir = f"{dataset}_{opt.init}_{opt.init_param}"
        return os.path.join(model_dir, subdir)


class BaseOptions(object):
    def __init__(self):
        # argparse utils

        def comp(type, op, ref):
            op = getattr(type, f"__{op}__")

            def check(value):
                ivalue = type(value)
                if not op(ivalue, ref):
                    raise argparse.ArgumentTypeError(f"expected value {op} {ref}, but got {value}")
                return ivalue

            return check

        def int_gt(i):
            return comp(int, "gt", i)

        def float_gt(i):
            return comp(float, "gt", i)

        pos_int = int_gt(0)
        nonneg_int = int_gt(-1)
        pos_float = float_gt(0)

        def get_unique_action_cls(actual_action_cls):
            class UniqueSetAttrAction(argparse.Action):
                def __init__(self, *args, **kwargs):
                    self.subaction = actual_action_cls(*args, **kwargs)

                def __call__(self, parser, namespace, values, option_string=None):
                    if isinstance(namespace, State.UniqueNamespace):
                        requires_unique = namespace.requires_unique()
                    else:
                        requires_unique = False
                    if requires_unique:
                        namespace.mark_set(self.subaction.dest, values)
                    self.subaction(parser, namespace, values, option_string)

                def __getattr__(self, name):
                    return getattr(self.subaction, name)

            return UniqueSetAttrAction

        self.parser = parser = argparse.ArgumentParser(description="PyTorch Dataset Distillation")

        action_registry = parser._registries["action"]
        for name, action_cls in action_registry.items():
            action_registry[name] = get_unique_action_cls(action_cls)

        parser.add_argument("--batch_size", type=pos_int, default=1024,
                            help="input batch size for training (default: 1024)")
        parser.add_argument("--test_batch_size", type=pos_int, default=1024,
                            help="input batch size for testing (default: 1024)")
        parser.add_argument("--epochs", type=pos_int, default=150, metavar="N",
                            help="number of total epochs to train (default: 150)")
        parser.add_argument("--decay_epochs", type=pos_int, default=50, metavar="N",
                            help="period of weight decay (default: 50)")
        parser.add_argument("--decay_factor", type=pos_float, default=0.1, metavar="N",
                            help="weight decay multiplicative factor (default: 0.1)")
        parser.add_argument("--lr", type=pos_float, default=0.01, metavar="LR",
                            help="learning rate (default: 0.01)")
        parser.add_argument("--init", type=str, default="xavier",
                            help="network initialization [normal|xavier|kaiming|orthogonal|zero|default]")
        parser.add_argument("--init_param", type=float, default=1.,
                            help="network initialization param: gain, std, etc.")
        parser.add_argument("--base_seed", type=int, default=1, metavar="S",
                            help="base random seed (default: 1)")
        parser.add_argument("--log_interval", type=int, default=100, metavar="N",
                            help="how many batches to wait before logging training status")
        parser.add_argument("--checkpoint_interval", type=int, default=10, metavar="N",
                            help="checkpoint interval (epoch)")
        parser.add_argument("--dataset", type=str, default="MNIST",
                            help="dataset: MNIST | MNIST_RGB | FASHION_MNIST | SVHN")
        parser.add_argument("--source_dataset", type=str, default=None,
                            help="dataset: MNIST | MNIST_RGB | FASHION_MNIST | SVHN")
        parser.add_argument("--forgetting_dataset", type=str, default=None,
                            help="dataset: MNIST | MNIST_RGB | FASHION_MNIST | SVHN")
        parser.add_argument("--mode", type=str, default="distill_basic",
                            help="mode: train | distill_basic | distill_adapt | forgetting ")
        parser.add_argument("--distill_lr", type=float, default=0.02,
                            help="learning rate to perform GD with distilled images PER STEP (default: 0.02)")
        parser.add_argument("--base_dir", type=str, default=None,
                            help="base_dir of run")
        parser.add_argument("--ipc", type=pos_int, default=1,
                            help="use #batch_size distilled images for each class in each step")
        parser.add_argument("--distill_steps", type=pos_int, default=10, help="Iterative distillation, use #num_steps * #batch_size * #classes distilled images."
                                 "See also --distill_epochs. The total number "
                                 "of steps is distill_steps * distill_epochs.")
        parser.add_argument("--distill_epochs", type=pos_int, default=3,
                            help="how many times to repeat all steps 1, 2, 3, 1, 2, 3, ...")
        parser.add_argument("--device_id", type=comp(int, "ge", -1), default=0, help="device id, -1 is cpu")
        parser.add_argument("--phase", type=str, default="train",
                            help="[train | test]")
        parser.add_argument("--num_workers", type=nonneg_int, default=8,
                            help="number of data loader workers")
        parser.add_argument("--log_level", type=str, default="INFO",
                            help="logging level, e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL")
        parser.add_argument("--expand_cls", action="store_true",
                            help="Expand the classifier when distill_basic and _adapt already finished to finetune a model on both datasets (dataset and source_dataset)")

    def get_state(self):
        if hasattr(self, "state"):
            return self.state

        logging.getLogger().setLevel(logging.DEBUG)
        self.opt, unknowns = self.parser.parse_known_args(namespace=State.UniqueNamespace())
        assert len(unknowns) == 0, f"Unexpected args: {unknowns}"
        self.state = State(self.opt)
        return self.set_state(self.state)

    def set_state(self, state):
        base_dir = state.get_base_directory()
        save_dir = state.get_save_directory()

        state.opt.start_time = time.strftime(r"%Y-%m-%d %H:%M:%S")

        utils.mkdir(save_dir)

        state.opt.log_file = os.path.join(state.get_save_directory(), "log.txt")

        state.opt.log_level = state.opt.log_level.upper()
        utils.logging.configure(state.opt.log_file, getattr(logging, state.opt.log_level))

        logging.info("=" * 40 + " " + state.opt.start_time + " " + "=" * 40)
        logging.info("Base directory is {}".format(base_dir))

        if state.phase == "test" and not os.path.isdir(base_dir):
            logging.warning("Base directory doesn't exist")

        _, state.opt.dataset_root, state.opt.nc, state.opt.input_size, state.opt.num_classes, \
            state.opt.dataset_normalization, state.opt.dataset_labels = datasets.get_info(state.dataset)

        # Write yaml
        yaml_str = yaml.dump(state.merge(public_only=True), default_flow_style=False, indent=4)
        logging.info("Options:\n\t" + yaml_str.replace("\n", "\n\t"))

        yaml_name = os.path.join(save_dir, "opt.yaml")
        if os.path.isfile(yaml_name):
            old_opt_dir = os.path.join(save_dir, "old_opts")
            utils.mkdir(old_opt_dir)
            with open(yaml_name, "r") as f:
                # ignore unknown ctors
                yaml.add_multi_constructor("", lambda loader, suffix, node: None)
                old_yaml = yaml.full_load(f)  # this is a dict
            old_yaml_time = old_yaml.get("start_time", "unknown_time")
            for c in ":-":
                old_yaml_time = old_yaml_time.replace(c, "_")
            old_yaml_time = old_yaml_time.replace(" ", "__")
            old_opt_new_name = os.path.join(old_opt_dir, "opt_{}.yaml".format(old_yaml_time))
            try:
                os.rename(yaml_name, old_opt_new_name)
                logging.warning(f"{yaml_name} already exists, moved to {old_opt_new_name}")
            except FileNotFoundError:
                logging.warning(f"{yaml_name} already exists, tried to move to {old_opt_new_name}, but failed, possibly due to other process having already done it")
                pass

            with open(yaml_name, "w") as f:
                f.write(yaml_str)

        # FROM HERE, we have saved options into yaml,
        #            can start assigning objects to opt, and
        #            modify the values for process-specific things
        # set device for CUDA or CPU = -1
        if state.device_id < 0:
            state.opt.device = torch.device("cpu")
        else:
            torch.cuda.set_device(state.device_id)
            state.opt.device = torch.device(f"cuda:{state.device_id}")

        if state.device.type == "cuda" and torch.backends.cudnn.enabled:
            torch.backends.cudnn.benchmark = True

        state.opt.seed = state.base_seed

        opt_dict = vars(self.parser.parse_args())
        opt_dict.pop("device_id")  # don"t compare this

        train_dataset = datasets.get_dataset("train", state.dataset)
        test_dataset = datasets.get_dataset("test", state.dataset)         

        state.opt.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=state.batch_size,
            num_workers=state.num_workers, pin_memory=True, shuffle=True)

        state.opt.test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=state.test_batch_size,
            num_workers=state.num_workers, pin_memory=True, shuffle=True)

        if state.forgetting_dataset != None:
            f_train_dataset = datasets.get_dataset("train", state.forgetting_dataset)
            f_test_dataset = datasets.get_dataset("test", state.forgetting_dataset)
            state.opt.f_train_loader = torch.utils.data.DataLoader(
                f_train_dataset, batch_size=state.batch_size,
                num_workers=state.num_workers, pin_memory=True, shuffle=True
                )
            state.opt.f_test_loader = torch.utils.data.DataLoader(
                f_test_dataset, batch_size=state.test_batch_size,
                num_workers=state.num_workers, pin_memory=True, shuffle=True
                )

        if state.source_dataset != None:
            test_dataset_old_domain = datasets.get_dataset('test', state.source_dataset)
            state.opt.source_test_loader = torch.utils.data.DataLoader(
                test_dataset_old_domain, batch_size=state.batch_size,
                num_workers=state.num_workers, pin_memory=True, shuffle=True
                )                            

        logging.info(f"train dataset size:\t{len(train_dataset)}")
        logging.info(f"test dataset size: \t{len(test_dataset)}")
        logging.info("datasets built!")

        return state


options = BaseOptions()
