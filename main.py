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
from torch.utils.data import Dataset

from networks.networks import LeNet
from train_distilled_image import distill
import utils
from base_options import options
from basics import evaluate_model, evaluate_steps
from utils.io import load_results, save_test_results
import datasets



class TensorDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]

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

    steps[1] = torch.add(steps[1], 10)

    return steps



def train_mode(state, train_loader=None, test_loader=None, source_test_loader=None, lrs=None):
    """
    Function to train a (LeNet-)model

    state: State-object
    train_loader: Dataloader for training, if None state.train_loader
    test_loader: Dataloader for testing, if None state.test_loader

    source_test_loader: Needed for evaluation in expand mode
    lrs: tensor/list of lrs

    return: changes model in state.models
    """

    model_dir = state.get_model_dir()
    utils.mkdir(model_dir)

    if train_loader == None:
        train_loader = state.train_loader
    if test_loader == None:
        test_loader = state.test_loader

    
    lr = state.lr
    if lrs != None:
        lr = lrs[0]

    optimizer = optim.SGD(state.models.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=state.epochs, gamma=state.decay_factor)
    if state.mode == "train":
        optimizer = optim.Adam(state.models.parameters(), lr=lr, betas=(0.5, 0.999))
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

    def epoch_fn_expanded():
        """
        Subfunction of training in expanded mode, gets called every epoch to train the model and evaluate it
        """
        
        state.models.train()
        train_iter = iter(train_loader)
        N = len(train_iter)

        for data, target in train_iter:

            # Set lr for the second half of epoch
            if N // 2 == len(train_iter):
                for g in optim.param_groups:
                    g['lr'] = lr[1]


            data, target = data.to(state.device, non_blocking=True), target.to(state.device, non_blocking=True)
            optimizer.zero_grad()
            output = state.models(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            # Reset lr for the first half of next epoch
            if len(train_iter) == 0:
                for g in optim.param_groups:
                    g['lr'] = lr[0]

        acc, loss = evaluate_model(state, state.models, test_loader_iter=iter(test_loader))
        acc_source, loss_source = evaluate_model(state, state.models, test_loader_iter=iter(source_test_loader))
        logging.info(
            f"Epoch: {epoch:>4}\tTest Accuracy {state.dataset}: {acc:.2%}\tTest Loss {state.dataset}: {loss:.4f}\n"\
            + f"{' ':>11}\tTest Accuracy {state.source_dataset}: {acc_source:.2%}\tTest Loss {state.source_dataset}: {loss_source:.4f}"
            )
    
    for epoch in range(state.epochs):
        if state.expand_cls:
            assert source_test_loader != None, "Please set a source test loader in expanded mode"
            epoch_fn_expanded()
        else:
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
            logging.info(f"\n{log_info} for {state.forgetting_dataset}:\nTest Accuracy: {acc:.2%}\tTest Loss: {loss:.4f}\n")

            acc_old, loss_old = evaluate_model(state, state.models, test_loader_iter=iter(state.test_loader))
            logging.info(f"\n{log_info} for {state.dataset}:\nTest Accuracy: {acc_old:.2%}\tTest Loss: {loss_old:.4f}\n")


        evaluate_forgetting("Evaluation BEFORE forgetting")
        train_mode(state, state.f_train_loader, state.f_test_loader)
        evaluate_forgetting("Evaluation AFTER forgetting")

    elif state.mode == "distill_basic":      

        if state.phase == "train":

            steps = distill(state, state.models)
            evaluate_steps(state, steps, f"Final evaluation for {state.dataset}")

        elif state.phase == "test":

            loaded_steps = list(load_results(state, device=state.device)[-1])
            my_dataset = TensorDataset(loaded_steps[0], loaded_steps[1])
            logging.info(f"Custom dataset length: {len(my_dataset)}")
            state.train_loader = torch.utils.data.DataLoader(my_dataset, batch_size=len(my_dataset), num_workers=0)

            acc, loss = evaluate_model(state, state.models, test_loader_iter=iter(state.test_loader))
            logging.info(f"Results for {state.dataset} BEFORE training with synthetic data:\nTest Accuracy: {acc:.2%}\tTest Loss: {loss:.4f}\n")

            train_mode(state, state.train_loader, state.test_loader, lrs=loaded_steps[2])

            acc, loss = evaluate_model(state, state.models, test_loader_iter=iter(state.test_loader))
            logging.info(f"Results for {state.dataset} AFTER training with synthetic data:\nTest Accuracy: {acc:.2%}\tTest Loss: {loss:.4f}\n")

        else:
            raise ValueError(f"phase: {state.phase}")

    elif state.mode == "distill_adapt":

        if state.phase == "train":

            test_string = f"Final evaluation for {{}} without expanded Classifier"
            steps = distill(state, state.models)
            evaluate_steps(state, steps, test_string.format(state.dataset))
            evaluate_steps(state, steps, test_string.format(state.source_dataset), test_loader=state.source_test_loader)

        elif state.phase == "test":

            loaded_steps = list(load_results(state, device=state.device)[-1])

            if state.expand_cls:
                new_steps = expand_model(state, loaded_steps, state.dataset)
                add_loaded_steps = list(load_results(state, mode="distill_basic", dataset=state.source_dataset, device=state.device)[-1])
                loaded_steps[0] = torch.cat((new_steps[0], add_loaded_steps[0]),0)
                loaded_steps[1] = torch.cat((new_steps[1], add_loaded_steps[1]),0)
                loaded_steps[2] = torch.cat((new_steps[2], add_loaded_steps[2]),0)

            my_dataset = TensorDataset(loaded_steps[0], loaded_steps[1])
            logging.info(f"Custom dataset length: {len(my_dataset)}")
            batch_size = len(my_dataset)
            if state.expand_cls:
                batch_size = len(my_dataset) // 2
            state.train_loader = torch.utils.data.DataLoader(my_dataset, batch_size=batch_size, num_workers=0)

            def evaluate_adapt(log_info: str) -> None:

                acc, loss = evaluate_model(state, state.models, test_loader_iter=iter(state.test_loader))
                logging.info(f"\n{log_info} for {state.dataset}:\nTest Accuracy: {acc:.2%}\tTest Loss: {loss:.4f}\n")

                acc_old, loss_old = evaluate_model(state, state.models, test_loader_iter=iter(state.source_test_loader))
                logging.info(f"\n{log_info} for {state.source_dataset}:\nTest Accuracy: {acc_old:.2%}\tTest Loss: {loss_old:.4f}\n")


            evaluate_adapt("Evaluation BEFORE adapting")
            train_mode(state, state.train_loader, state.test_loader, state.source_test_loader, loaded_steps[2])
            evaluate_adapt("Evaluation AFTER adapting")

        else:
            raise ValueError(f"phase: {state.phase}")

    else:
        raise NotImplementedError(f"unknown mode: {state.mode}")


if __name__ == "__main__":
    try:
        main(options.get_state())
    except Exception:
        raise
