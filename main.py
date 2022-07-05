from ast import Add
import logging
import random
import os
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.nn import init

from networks.networks import LeNet
from train_distilled_image import distill
import utils
from base_options import options
from basics import evaluate_model, evaluate_steps
from utils.io import load_results, save_test_results
import datasets



def permute_list(list):
    indices = np.random.permutation(len(list))
    return [list[i] for i in indices]

def expand_model(state, steps, dataset=None):
    """
    Expands the classifier of the current model for the num_classes of the given dataset
    and shifts the train and test dataloader labels

    state: State-object
    dataset: name of the dataset

    return: changes objects in the State-object
    """

    _, _, _, _, num_classes, _, _ = datasets.get_info(dataset)

    with state.models.unflatten_weight(state.models.get_param()):
        classifier = state.models.fc3
        new_in_features = classifier.in_features
        new_out_features = classifier.out_features + num_classes
        bias_flag = False

        tmp_weights = classifier.weight.data.clone()
        if not isinstance(classifier.bias, type(None)):
            tmp_bias = classifier.bias.data.clone()
            bias_flag = True

        classifier = nn.Linear(new_in_features, new_out_features, bias=bias_flag)
        init.xavier_normal_(classifier.weight, gain=state.init_param)
        classifier.to(state.device)

        # copy back the temporarily saved parameters for the slice of previously trained classes.
        classifier.weight.data[0:-num_classes, :] = tmp_weights
        if not isinstance(classifier.bias, type(None)):
            classifier.bias.data[0:-num_classes] = tmp_bias
        state.models.fc3 = classifier

    test_dataset = datasets.get_dataset("test", dataset)
    test_dataset.targets += num_classes
    state.test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=state.test_batch_size,
        num_workers=state.num_workers, pin_memory=True, shuffle=True
        )

    result_steps = []

    for (data, label, lr) in steps:
        result_steps.append((data, torch.add(label, 10), lr))
        
    return result_steps



def train_mode(state, train_loader=None, test_loader=None):
    """
    Function to train a (LeNet-)model

    state: State-object
    train_loader: Dataloader for training, if None state.train_loader
    test_loader: Dataloader for testing, if None state.test_loader

    return: changes model in state.models
    """

    model_dir = state.get_model_dir()
    utils.mkdir(model_dir)

    if train_loader == None:
        train_loader = state.train_loader
    if test_loader == None:
        test_loader = state.test_loader

    optimizer = optim.Adam(state.models.parameters(), lr=state.lr, betas=(0.5, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=state.decay_epochs, gamma=state.decay_factor)

    def epoch_fn():
        """
        Subfunction of training, gets called every epoch to train the model and evaluate it
        """
        
        state.models.train()

        for data, target in train_loader:
            data, target = data.to(state.device, non_blocking=True), target.to(state.device, non_blocking=True)
            optimizer.zero_grad()
            output = state.models(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

        acc, loss = evaluate_model(state, state.models, test_loader_iter=iter(test_loader))
        logging.info(f"Epoch: {epoch:>4}\tTest Accuracy: {acc:.2%}\tTest Loss: {loss:.4f}")
        scheduler.step()
    
    for epoch in range(state.epochs):
        epoch_fn()


def main(state):
    """
    Main method of the script. The function is divided in the state.mode(s) and in every mode is divided in state.phase(s)

    state: State object containing all values and objects needed for execution of the script

    return: void
    """

    # Preparation of the model and some help variables
    model_dir = state.get_model_dir()
    dataset = state.dataset

    if state.mode == "distill_adapt":
        dataset = state.source_dataset

    model_path = os.path.join(model_dir, f"LeNet_{dataset}")
    state.models = LeNet(state)

    if state.mode != "train" or state.phase != "train":
        state.models.load_state_dict(torch.load(model_path, map_location=state.device))

    if state.phase == "test":
        
        unique_data_label = loaded_steps = []

        def get_data_label():
            return unique_data_label

        def get_lrs():
            return tuple(s[-1] for s in loaded_steps)

        class StepCollection(object):
            def __init__(self, state):
                self.state = state
                self.steps = []
                for (data, label), lr in zip(get_data_label(), get_lrs()):
                    self.steps.append((data, label, lr))

            def get_steps(self):
                return self.steps
    
        class TestRunner(object):
            def __init__(self, state):
                self.state = state
                self.stepss = StepCollection(state)

            def run(self, message):
                steps = self.stepss.get_steps()
                res = []
                with self.__seed(self.state.seed):
                    res = evaluate_steps(
                        self.state, steps,
                        f"Final Test phase: ",
                        message
                        )
                return res

            @contextmanager
            def __seed(self, seed):
                cpu_rng = torch.get_rng_state()
                cuda_rng = torch.cuda.get_rng_state(self.state.device)
                torch.random.default_generator.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                yield
                torch.set_rng_state(cpu_rng)
                torch.cuda.set_rng_state(cuda_rng, self.state.device)

            def num_steps(self):
                return self.state.distill_steps * self.distill_epochs

    # Train mode: Train a (LeNet-)model for a given dataset and saving it or testing it
    if state.mode == "train":

        if state.phase == "train":
            state.models.reset(state)
            train_mode(state)
            torch.save(state.models.state_dict(), model_path)

        elif state.phase == "test":
            pass

        else:
            raise ValueError(f"phase: {state.phase}")

        acc, loss = evaluate_model(state, state.models, test_loader_iter=iter(state.test_loader))
        logging.info(f"Final Test Accuracy: {acc:.2%}\tFinal Test Loss: {loss:.4f}")

    # Forgetting mode: Validate model by learning a different dataset
    elif state.mode == "forgetting":

        def evaluate_forgetting(log_info: str) -> None:

            acc, loss = evaluate_model(state, state.models, test_loader_iter=iter(state.f_test_loader))
            logging.info(f"\n{log_info} for {state.forget_dataset}:\nTest Accuracy: {acc:.2%}\tTest Loss: {loss:.4f}\n")

            acc_old, loss_old = evaluate_model(state, state.models, test_loader_iter=iter(state.test_loader))
            logging.info(f"\n{log_info} for {state.dataset}:\nTest Accuracy: {acc_old:.2%}\tTest Loss: {loss_old:.4f}\n")


        evaluate_forgetting("Evaluation before forgetting")
        train_mode(state, state.f_train_loader, state.f_test_loader)
        evaluate_forgetting("Evaluation after forgetting")

    elif state.mode == "distill_basic":      

        if state.phase == "train":

            steps = distill(state, state.models)
            evaluate_steps(state, steps, f"Final evaluation for {state.dataset}")

        elif state.phase == "test":

            loaded_steps = load_results(state, device=state.device)
            unique_data_label = [s[:-1] for s in loaded_steps]

            # run tests
            test_runner = TestRunner(state)
            res = test_runner.run(f"Final evaluation for {state.dataset}")
            save_test_results(state, res)

        else:
            raise ValueError(f"phase: {state.phase}")

    elif state.mode == "distill_adapt":

        if state.phase == "train":

            test_string = f"Final evaluation for {{}} without expanded Classifier"
            steps = distill(state, state.models)
            evaluate_steps(state, steps, test_string.format(state.dataset))
            evaluate_steps(state, steps, test_string.format(state.source_dataset), test_loader=state.source_test_loader)

        elif state.phase == "test":

            loaded_steps = load_results(state, device=state.device)
            if state.expand_cls:
                loaded_steps = expand_model(state, loaded_steps, state.dataset)
                add_loaded_steps = load_results(state, mode="distill_basic", dataset=state.source_dataset, device=state.device)
                loaded_steps =  add_loaded_steps + loaded_steps
            unique_data_label = [s[:-1] for s in loaded_steps]

            # run tests
            test_runner = TestRunner(state)
            res = test_runner.run(f"Final evaluation for {state.dataset} and {state.source_dataset}")
            save_test_results(state, res)

        else:
            raise ValueError(f"phase: {state.phase}")

    else:
        raise NotImplementedError(f"unknown mode: {state.mode}")


if __name__ == "__main__":
    try:
        main(options.get_state())
    except Exception:
        raise
