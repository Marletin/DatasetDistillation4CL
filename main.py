import logging
import os
from tracemalloc import start
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import time

from torch.nn import init
from torch.utils.data import Dataset

import datasets
import utils

from networks.networks import LeNet
from train_distilled_image import distill
from base_options import options
from basics import evaluate_model, evaluate_steps
from utils.io import load_results




class TensorDatasetAdapt(Dataset):
    def __init__(self, images, labels, datasets):
        self.images = images.detach().float()
        self.labels = labels.detach()
        self.datasets = datasets.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index], self.datasets[index]

    def __len__(self):
        return self.images.shape[0]


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


def load_synthetic_images_last(state, dataset=None, batch_size=None, shuffle=False):
    """
    Loading the synthetic images from the last distill step and the last distill epoch

    state: state-object

    mode: needed if synthetic images should be loaded from a different mode than state.mode
    dataset: needed if synthetic images should be loaded from a different dataset than state.dataset
        If you provide a mode you need to provide a dataset and vice versa
    batch_size: Batch size for the DataLoader
    shuffle: Boolean if the DataLoader should shuffle the data

    return: Synthetic Learning Rates
    """
    loaded_steps = list(load_results(state, dataset, state.device)[-1])
    
    class_descriptor = []
    for _ in loaded_steps[1]:
        class_descriptor.append(0)
    class_desc_tensor = torch.FloatTensor(class_descriptor)
    
    my_dataset = TensorDatasetAdapt(loaded_steps[0], loaded_steps[1], class_desc_tensor)
    dataset_len = len(my_dataset)
    batch_size = batch_size or dataset_len
    logging.info(f"Custom dataset length: {dataset_len}")
    state.train_loader = torch.utils.data.DataLoader(my_dataset, batch_size=batch_size, num_workers=0, shuffle=shuffle)
    return loaded_steps[2]


def load_synthetic_images_all(state, mode=None, dataset=None, batch_size=None, shuffle=False):
    steps = list(load_results(state, mode, dataset, state.device))
    data_list, label_list, lr_list = [], [], []

    for (data, labels, lr) in steps:
        for data_point in data:
            data_list.append(data_point)
        for label in labels:
            label_list.append(label)
        lr_list.append(lr)

    label_tensor = torch.stack(label_list)
    data_tensor = torch.stack(data_list)

    my_dataset = TensorDataset(data_tensor, label_tensor)
    dataset_len = len(my_dataset)
    batch_size = batch_size or dataset_len
    logging.info(f"Custom dataset length: {dataset_len}")
    state.train_loader = torch.utils.data.DataLoader(my_dataset, batch_size=batch_size, num_workers=0, shuffle=shuffle)
    
    return lr_list


def load_synthetic_images_shuffle_last(state, dataset, batch_size=None, shuffle=True):
    """
    Loading the synthetic images from two datasets from the last distill step and the last distill epoch and combine them in one DataLoader

    state: state-object

    mode: needed if synthetic images should be loaded from a different mode than state.mode
    dataset: needed if synthetic images should be loaded from a different dataset than state.dataset

    batch_size: Batch size for the DataLoader
    shuffle: Boolean if the DataLoader should shuffle the data
    """
    loaded_steps = list(load_results(state, device=state.device)[-1])
    add_loaded_steps = list(load_results(state, dataset, device=state.device)[-1])

    class_descriptor = []
    for _ in loaded_steps[1]:
        class_descriptor.append(0)
    for _ in add_loaded_steps[1]:
        class_descriptor.append(1)
        
    class_desc_tensor = torch.FloatTensor(class_descriptor)


    loaded_steps[0] = torch.cat((loaded_steps[0], add_loaded_steps[0]),0)
    loaded_steps[1] = torch.cat((loaded_steps[1], add_loaded_steps[1]),0)
    loaded_steps[2] = torch.cat((loaded_steps[2], add_loaded_steps[2]),0)
    
    print(*loaded_steps[1], *loaded_steps[2], class_desc_tensor)
    
    my_dataset = TensorDatasetAdapt(loaded_steps[0], loaded_steps[1], class_desc_tensor)
    dataset_len = len(my_dataset)
    batch_size = batch_size or dataset_len
    logging.info(f"Custom dataset length: {dataset_len}")
    state.train_loader = torch.utils.data.DataLoader(my_dataset, batch_size=batch_size, num_workers=0, shuffle=shuffle)
    return loaded_steps[2]

def load_synthetic_images_shuffle_expand(state, mode, dataset, batch_size=None, shuffle=True):
    """
    Loading the synthetic images from two datasets and combine them in one DataLoader

    state: state-object

    mode: needed if synthetic images should be loaded from a different mode than state.mode
    dataset: needed if synthetic images should be loaded from a different dataset than state.dataset

    batch_size: Batch size for the DataLoader
    shuffle: Boolean if the DataLoader should shuffle the data
    """
    loaded_steps = list(load_results(state, state.device)[-1])
    new_steps = expand_model(state, loaded_steps, state.dataset)
    add_loaded_steps = list(load_synthetic_images_last(state, mode, dataset, device=state.device)[-1])
    size_data1 = len(new_steps[0])
    size_data2 = len(add_loaded_steps[0])
    loaded_steps[0] = torch.cat((new_steps[0], add_loaded_steps[0]),0)
    loaded_steps[1] = torch.cat((new_steps[1], add_loaded_steps[1]),0)
    loaded_steps[2] = torch.cat((new_steps[2].repeat(size_data1), add_loaded_steps[2].repeat(size_data2)),0)
    loaded_steps.append(torch.cat((torch.tensor([0]).repeat(size_data1), torch.tensor([1]).repeat(size_data2)),0))
    my_dataset = TensorDatasetAdapt(loaded_steps[0], loaded_steps[1], loaded_steps[2], loaded_steps[3])
    dataset_len = len(my_dataset)
    batch_size = batch_size or dataset_len
    logging.info(f"Custom dataset length: {dataset_len}")
    state.train_loader = torch.utils.data.DataLoader(my_dataset, batch_size=batch_size, num_workers=0, shuffle=shuffle)


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

    acc_list = []
    loss_list = []
    acc_list_origin = []

    train_loader = train_loader or state.train_loader
    test_loader = test_loader or state.test_loader
    lr = state.lr
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
        acc_list.append(acc)
        loss_list.append(loss)
        logging.info(f"Epoch: {epoch:>4}\tTest Accuracy: {acc:.2%}\tTest Loss: {loss:.4f}")
        if state.mode == "forgetting":
            acc2, loss2 = evaluate_model(state, state.models, test_loader_iter=iter(state.test_loader))
            logging.info(f"Source: {state.dataset}\tTest Accuracy: {acc2:.2%}\tTest Loss: {loss2:.4f}")
            acc_list_origin.append(acc2)
        scheduler.step()

    for epoch in range(state.epochs):
        epoch_fn()

    logging.info(f"List of accuracies: {acc_list}")
    logging.info(f"List of losses: {loss_list}")
    if state.mode == "forgetting":
        logging.info(f"List of source accuracies: {acc_list_origin}")

def train_mode_adapt(state, train_loader=None, test_loader=None, source_test_loader=None, lrs=None):
    """
    Function to train a (LeNet-)model

    state: State-object
    train_loader: Dataloader for training, if None state.train_loader
    test_loader: Dataloader for testing, if None state.test_loader

    source_test_loader: Needed for evaluation in expand mode
    lrs: tensor/list of lrs

    return: changes model in state.models
    """
    train_loader = train_loader or state.train_loader
    test_loader = test_loader or state.test_loader
    accuracies_tune = []
    accuracies_origin = []

    lr = state.lr
    optimizer = optim.SGD(state.models.parameters(), lr=lr)
    
    if lrs != None:
        optimizer = optim.SGD(state.models.parameters(), lr=1)
        logging.info(f"Using synthetic lrs: {lrs}")
    else:
        logging.info(f"Leraning Rate is not synthetic: {lr}")
    
    def epoch_fn_syn_lr():
        """
        Subfunction of training in expanded mode, gets called every epoch to train the model and evaluate it
        """
        print("!!!!!!!!!!!!!!!!!!!!!!!!")
        state.models.train()

        for data, target, class_descriptor in train_loader:
            data, target = data.to(state.device, non_blocking=True), target.to(state.device, non_blocking=True)
           
            weights = []
            for index in class_descriptor:
                weights.append(float(lrs[int(index)]))
            
            weights = torch.FloatTensor(weights)
            weights = weights.to(state.device, non_blocking=True)
            
            optimizer.zero_grad()
            output = state.models(data)
            loss = F.cross_entropy(output, target, reduction="none")            
            loss = loss * weights
            loss = loss.sum() / weights.sum()
            loss.backward()
            optimizer.step()

        acc, loss = evaluate_model(state, state.models, test_loader_iter=iter(test_loader))
        acc_source, loss_source = evaluate_model(state, state.models, test_loader_iter=iter(source_test_loader))
        accuracies_tune.append(float(f"{acc:.4f}"))
        accuracies_origin.append(float(f"{acc_source:.4f}"))
        logging.info(
            f"Epoch: {epoch:>4}\tTest Accuracy {state.dataset}: {acc:.2%}\tTest Loss {state.dataset}: {loss:.4f}\n"\
            + f"{' ':>11}\tTest Accuracy {state.source_dataset}: {acc_source:.2%}\tTest Loss {state.source_dataset}: {loss_source:.4f}"
            )

    def epoch_fn():
        """
        Subfunction of training in expanded mode, gets called every epoch to train the model and evaluate it
        """

        state.models.train()

        for data, target, _ in train_loader:
            data, target = data.to(state.device, non_blocking=True), target.to(state.device, non_blocking=True)

            optimizer.zero_grad()
            output = state.models(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            

        acc, loss = evaluate_model(state, state.models, test_loader_iter=iter(test_loader))
        acc_source, loss_source = evaluate_model(state, state.models, test_loader_iter=iter(source_test_loader))
        accuracies_tune.append(float(f"{acc:.4f}"))
        accuracies_origin.append(float(f"{acc_source:.4f}"))
        logging.info(
            f"Epoch: {epoch:>4}\tTest Accuracy {state.dataset}: {acc:.2%}\tTest Loss {state.dataset}: {loss:.4f}\n"\
            + f"{' ':>11}\tTest Accuracy {state.source_dataset}: {acc_source:.2%}\tTest Loss {state.source_dataset}: {loss_source:.4f}"
            )
    
    for epoch in range(state.epochs):
        if lrs == None:
            epoch_fn()
        else:
            epoch_fn_syn_lr()
    
    logging.info(f"Accuracies for {state.dataset}\n{accuracies_tune}")
    logging.info(f"Accuracies for {state.source_dataset}\n{accuracies_origin}")
    logging.info(f"Zipped Accuracies: {list(zip(accuracies_tune, accuracies_origin))}")

##########################################################################################################################
def main(state):
    """
    Main method of the script. The function is divided in the state.mode(s) and in every mode is divided in state.phase(s)

    state: State object containing all values and objects needed for execution of the script
    """

    # Preparation of the model and some help variables
    start_time = time.process_time()
    model_dir = state.get_model_dir()
    utils.mkdir(model_dir)
    dataset = state.dataset

    if state.mode == "distill_adapt":
        dataset = state.source_dataset

    model_path = os.path.join(model_dir, f"LeNet_{dataset}")
    state.models = LeNet(state)

    # Loading model if not in training mode or training phase (for loading model in testing phase of training mode)
    if state.mode != "train" or state.phase != "train":
        state.models.load_state_dict(torch.load(model_path, map_location=state.device))

    # Train mode: Train a (LeNet-)model for a given dataset and saving it or testing it
    if state.mode == "train":

        if state.phase == "train":
            state.models.reset(state)
            # Comment out next line too save random initialized model
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
            logging.info(f"\n{state.forgetting_dataset}: {log_info}:\nTest Accuracy: {acc:.2%}\tTest Loss: {loss:.4f}\n")

            acc_old, loss_old = evaluate_model(state, state.models, test_loader_iter=iter(state.test_loader))
            logging.info(f"\n{state.dataset}: {log_info}:\nTest Accuracy: {acc_old:.2%}\tTest Loss: {loss_old:.4f}\n")


        evaluate_forgetting("Evaluation BEFORE forgetting")
        train_mode(state, state.f_train_loader, state.f_test_loader)
        evaluate_forgetting("Evaluation AFTER forgetting")

    elif state.mode == "distill_basic":      

        if state.phase == "train":

            steps = distill(state, state.models)
            evaluate_steps(state, steps, f"Final evaluation for {state.dataset}")

        elif state.phase == "test":

            lrs = load_synthetic_images_last(state)

            acc, loss = evaluate_model(state, state.models, test_loader_iter=iter(state.test_loader))
            logging.info(f"Results for {state.dataset} BEFORE training with synthetic data:\nTest Accuracy: {acc:.2%}\tTest Loss: {loss:.4f}\n")

            train_mode(state, state.train_loader, state.test_loader, lrs=lrs)

            acc, loss = evaluate_model(state, state.models, test_loader_iter=iter(state.test_loader))
            logging.info(f"Results for {state.dataset} AFTER training with synthetic data:\nTest Accuracy: {acc:.2%}\tTest Loss: {loss:.4f}\n")

        else:
            raise ValueError(f"phase: {state.phase}")

    elif state.mode == "distill_adapt":

        if state.phase == "train":

            test_string = f"Final evaluation for {{}}"
            steps = distill(state, state.models)
            evaluate_steps(state, steps, test_string.format(state.dataset))
            evaluate_steps(state, steps, test_string.format(state.source_dataset), test_loader=state.source_test_loader)

        elif state.phase == "test":

            # lrs = load_synthetic_images_last(state)
            # lrs = load_synthetic_images_all(state)
            lrs = load_synthetic_images_shuffle_last(state, state.source_dataset)
            
            # state.lr = float(lrs[1])
            # state.lr = (float(lrs[1]) + float(lrs[0])) / 2

            def evaluate_adapt(log_info: str) -> None:

                acc, loss = evaluate_model(state, state.models, test_loader_iter=iter(state.test_loader))
                logging.info(f"\n{state.dataset}: {log_info}:\nTest Accuracy: {acc:.2%}\tTest Loss: {loss:.4f}\n")

                acc_old, loss_old = evaluate_model(state, state.models, test_loader_iter=iter(state.source_test_loader))
                logging.info(f"\n{state.source_dataset}: {log_info}:\nTest Accuracy: {acc_old:.2%}\tTest Loss: {loss_old:.4f}\n")


            evaluate_adapt("Evaluation BEFORE adapting")
            # train_mode(state, state.train_loader, state.test_loader)
            logging.info(lrs)
            train_mode_adapt(state, state.train_loader, state.test_loader, state.source_test_loader)
            evaluate_adapt("Evaluation AFTER adapting")

        else:
            raise ValueError(f"phase: {state.phase}")

    else:
        raise NotImplementedError(f"unknown mode: {state.mode}")

    end_time = time.process_time()
    res_time = (end_time - start_time) / 60
    logging.info(f"CPU Time: {res_time:.2f} minutes")
##########################################################################################################################

if __name__ == "__main__":
    try:
        main(options.get_state())
    except Exception:
        raise
