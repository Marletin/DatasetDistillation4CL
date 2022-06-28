from __future__ import print_function

import logging
import os
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from networks.networks import LeNet
from train_distilled_image import distill
import utils
from base_options import options
from basics import evaluate_model, evaluate_steps
from utils.io import load_results, save_test_results


def epoch_fn(state, model, epoch, optimizer, train_loader, test_loader):

    model.train()

    for data, target in train_loader:
        data, target = data.to(state.device, non_blocking=True), target.to(state.device, non_blocking=True)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

    acc, loss = evaluate_model(state, model, test_loader_iter=state.test_loader)
    logging.info(f"Epoch: {epoch:>4}\tAccuracy: {acc[0]:.2%}\tLoss: {loss[0]:.4f}")
 
def train_mode(state, train_loader = None, test_loader=None):
    
    model_dir = state.get_model_dir()
    utils.mkdir(model_dir)

    if train_loader == None:
        train_loader = state.train_loader
    if test_loader == None:
        testloader = state.test_loader

    model = LeNet(state)
    model.reset(state)

    optimizer = optim.Adam(model.parameters(), lr=state.lr, betas=(0.5, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=state.decay_epochs, gamma=state.decay_factor)
    for epoch in range(state.epochs):
        epoch_fn(state, model, epoch, optimizer, train_loader, test_loader)
        scheduler.step()
    
    return model

def main(state):
    model_dir = state.get_model_dir()
    dataset = state.dataset
    if state.mode == "distill_adapt":
        dataset = state.source_dataset
    model_path = os.path.join(model_dir, f"net_{dataset}")

    if state.mode == "train":
        if state.phase == "train":
            model = train_mode(state)
            torch.save(model.state_dict(), model_path)
        elif state.phase == "test":
            model = LeNet(state)
            model.load_state_dict(torch.load(model_path, map_location=state.device))
        else:
            raise ValueError(f"phase: {state.phase}")
        acc, loss = evaluate_model(state, model, test_loader_iter=state.test_loader)
        logging.info(f"Accuracy: {acc[0]:.2%}\tLoss: {loss[0]:.4f}")

    elif state.mode in ["forgetting"]:
        model = train_mode(state, state.f_train_loader)
        acc, loss = evaluate_model(state, model, test_loader_iter=state.f_test_loader)
        acc_old, loss_old = evaluate_model(state, model, test_loader_iter=state.test_loader)
        logging.info(f"Stats for {state.forget_dataset}:\nAccuracy: {acc[0]:.2%}\tLoss: {loss[0]:.4f}\n\nStats for {state.dataset}:\nAccuracy: {acc_old[0]:.2%}\tLoss: {loss_old[0]:.4f}")

    elif state.mode in ["distill_basic", "distill_adapt"]:

        state.models = LeNet(state)
        state.models.load_state_dict(torch.load(model_path, map_location=state.device))

        if state.mode == "distill_adapt":
            with state.models.unflatten_weight(state.models.get_param()):
                classifier = state.models.fc3
                new_in_features = classifier.in_features
                new_out_features = classifier.out_features + state.num_classes
                bias_flag = False

                tmp_weights = classifier.weight.data.clone()
                if not isinstance(classifier.bias, type(None)):
                    tmp_bias = classifier.bias.data.clone()
                    bias_flag = True

                classifier = nn.Linear(new_in_features, new_out_features, bias=bias_flag)
                classifier.to(state.device)

                # copy back the temporarily saved parameters for the slice of previously trained classes.
                classifier.weight.data[0:-state.num_classes, :] = tmp_weights
                if not isinstance(classifier.bias, type(None)):
                    classifier.bias.data[0:-state.num_classes] = tmp_bias
                state.models.fc3 = classifier

      

        if state.phase == "train":
            steps = distill(state, state.models)
            evaluate_steps(state, steps, f"Final evaluation for {state.dataset}")
            if state.mode == "distill_adapt":
                evaluate_steps(state, steps, f"Final evaluation for {state.source_dataset}", test_loader=state.source_test_loader)

        elif state.phase == "test":

            loaded_steps = load_results(state, device=state.device)

            unique_data_label = [s[:-1] for s in loaded_steps[:state.distill_steps]]

            def get_data_label(state):
                return [x for _ in range(state.distill_epochs) for x in unique_data_label]

            def get_lrs():
                return tuple(s[-1] for s in loaded_steps)

            class StepCollection(object):
                def __init__(self, state):
                    self.state = state
                    self.steps = []
                    for (data, label), lr in zip(get_data_label(self.state), get_lrs()):
                        self.steps.append((data, label, lr))

                def get_steps(self):
                    return self.steps
        
            class TestRunner(object):
                def __init__(self, state):
                    self.state = state
                    if state.test_distill_epochs is None:
                        self.test_distill_epochs = state.distill_epochs
                    else:
                        self.test_distill_epochs = state.test_distill_epochs
                    with state.pretend(distill_epochs=self.test_distill_epochs):
                        self.stepss = StepCollection(state)

                def run(self):
                    with self.state.pretend(distill_epochs=self.test_distill_epochs):
                        steps = self.stepss.get_steps()
                        res = []
                        with self.__seed(self.state.seed + 2):
                            res = evaluate_steps(
                                self.state, steps,
                                f"Test phase: ",
                                f"Final evaluation for {state.dataset} and {state.source_dataset}"
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
                    return self.state.distill_steps * self.test_distill_epochs

            # run tests
            test_runner = TestRunner(state)
            res = test_runner.run()
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
