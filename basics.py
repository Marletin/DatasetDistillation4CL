import logging
import math

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils import pm

def train_steps_inplace(model, steps, callback):
    """
    Train model params without changing the model. Usefull to evaluate the synthetic data during the distillation process.
    model: model to train on
    steps: triple -> data, label, lr
    callback: called to evaluate progress (steps are set in evaluate_steps() -> test_at_steps)

    return: trained params
    """
    params = model.get_param()
    for i, (data, label, lr) in enumerate(steps):
        if callback is not None:
            callback(i, params)

        data = data.detach()
        label = label.detach()
        lr = lr.detach()
        model.train()

        output = model.forward_with_param(data, params)
        loss = F.cross_entropy(output, label)
        loss.backward(lr.squeeze())

        with torch.no_grad():
            params.sub_(params.grad)
            params.grad = None

    if callback is not None:
        callback(len(steps), params)

    return params


def evaluate_model(state, model, params=None, test_loader_iter=None):
    device = state.device
    corrects = losses = total = 0

    if test_loader_iter is None:
        test_loader_iter = iter(state.test_loader)

    model.eval()

    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader_iter):
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

            if params is None:
                output = model(data)
            else:
                output = model.forward_with_param(data, params)

            pred = output.argmax(1)  # get the index of the max log-probability

            correct_list = pred == target
            losses += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            corrects += correct_list.sum().item()
            total += output.size(0)

        losses /= total

    accs = corrects / total
    
    return accs, losses


def fixed_width_fmt(num):
    if math.isnan(num):
        return f'{str(num):>6}'
    return f'{num:>6.6f}'


def _desc_step(steps, i):
    if i == 0:
        return "before steps"
    lr = steps[i - 1][-1]
    return f'step {i:2d} (lr={fixed_width_fmt(lr.sum().item())})'


def format_stepwise_results(steps, info, res):
    accs = res[1]
    losses = res[2]

    def format_into_line(*fields, align='>'):
        single_fmt = '{{:{}24}}'.format(align)
        return ' '.join(single_fmt.format(f) for f in fields)

    msgs = [format_into_line('STEP', 'ACCURACY', 'LOSS', align='^')]
    acc_fmt = f'{{: >8.2%}}'
    loss_fmt = f'{{: >8.4f}}'
    for at_step, acc, loss in zip(res[0], accs, losses):

        desc = _desc_step(steps, at_step)
        loss_str = loss_fmt.format(loss)
        acc_str = acc_fmt.format(acc)
        msgs.append(format_into_line(desc, acc_str, loss_str))

    return '{} test results:\n{}'.format(info, '\n'.join(('\t' + m) for m in msgs))


def evaluate_steps(state, steps, prefix, details='', test_loader=None):
    model = state.models
    n_steps = len(steps)
    test_at_steps = [0, n_steps // 2, n_steps]
    N = len(test_at_steps)

    if test_loader == None:
        test_loader = state.test_loader

    test_nets_desc = f"evaluate {N} steps: {sorted(test_at_steps)}"

    def _evaluate_steps(comment):

        pbar_desc = f"{prefix} ({comment})"
        pbar = tqdm(total=N, desc=pbar_desc)

        at_steps = []
        accs = []
        losses = []
        losses_old_dom = []
        accs_old_dom = []

        params = model.get_param(clone=True)

        def test_callback(at_step, params):
            if at_step not in test_at_steps:
                return

            acc, loss = evaluate_model(state, model, params,test_loader_iter=iter(test_loader))

            if state.mode == "distill_adapt":
                acc_old_dom, loss_old_dom = evaluate_model(state, model, params, test_loader_iter=iter(state.source_test_loader))
                losses_old_dom.append(loss_old_dom)
                accs_old_dom.append(acc_old_dom)
            

            at_steps.append(at_step)
            accs.append(acc)
            losses.append(loss)
            pbar.update()

        train_steps_inplace(model, steps, test_callback)

        pbar.close()

        at_steps = torch.as_tensor(np.array(at_steps), device=state.device)  # STEP
        accs = torch.as_tensor(np.array(accs), device=state.device)          # STEP x MODEL (x CLASS)
        losses = torch.as_tensor(np.array(losses), device=state.device)      # STEP x MODEL
        accs_old_dom = torch.as_tensor(np.array(accs_old_dom), device=state.device)          
        losses_old_dom = torch.as_tensor(np.array(losses_old_dom), device=state.device) 
        return [at_steps, accs, losses, accs_old_dom, losses_old_dom]

    logging.info('')
    logging.info(f'{prefix} {details}:')

    res = _evaluate_steps(test_nets_desc)
    result_title = f'{prefix} Final evaluation for {state.dataset} ({test_nets_desc})'

    logging.info(format_stepwise_results(steps, result_title, res[:3]))
    logging.info('')

    if state.mode == "distill_adapt" and state.phase == "test":
        new_res = [res[0], res[3], res[4]]
        result_title = f'{prefix} Final evaluation for {state.source_dataset} ({test_nets_desc})'

        logging.info(format_stepwise_results(steps, result_title, new_res))

    return res
