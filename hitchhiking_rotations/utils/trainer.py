#
# Copyright (c) 2024, MPI-IS, Jonas Frey, Rene Geist, Mikel Zhobro.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
import torch


class EarlyStopper:
    def __init__(self, model, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")
        self.early_stopped = False
        self.model = model
        self.best_state_dict = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.best_state_dict = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stopped = True
                return True
        return False


class Trainer:
    def __init__(
        self,
        preprocess_target,
        preprocess_input,
        postprocess_pred_loss,
        postprocess_pred_logging,
        loss,
        model,
        lr,
        optimizer,
        logger,
        verbose,
        device,
    ):
        self.preprocess_target = preprocess_target
        self.preprocess_input = preprocess_input
        self.postprocess_pred_loss = postprocess_pred_loss
        self.postprocess_pred_logging = postprocess_pred_logging
        self.loss = loss
        self.model = model
        self.logger = logger
        self.verbose = verbose
        self.device = device

        self.model.to(device)

        if optimizer == "SGD":
            self.opt = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        elif optimizer == "Adam":
            self.opt = torch.optim.Adam(self.model.parameters(), lr=lr, eps=1e-7, amsgrad=False)
        elif optimizer == "AdamW":
            self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr)

        self.reset()

        self.nr_training_steps = 0
        self.nr_test_steps = 0

        self.early_stopper = EarlyStopper(model=self.model, patience=10, min_delta=0)

    def train_batch(self, x, target, epoch):
        self.model.train()
        self.opt.zero_grad()

        with torch.no_grad():
            pp_target = self.preprocess_target(target)

        x = self.preprocess_input(x)
        pred = self.model(x)
        pred_loss = self.postprocess_pred_loss(pred)

        loss = self.loss(pred_loss, pp_target)
        loss.backward()
        self.opt.step()

        with torch.no_grad():
            pred_log = self.postprocess_pred_logging(pred)
            self.logger.log("train", epoch, pred_log, target, loss.item())
            self.nr_training_steps += 1

        return loss

    @torch.no_grad()
    def test_batch(self, x, target, epoch, mode):
        self.model.eval()
        x = self.preprocess_input(x)
        pred = self.model(x)
        pred_loss = self.postprocess_pred_loss(pred)
        pp_target = self.preprocess_target(target)
        loss = self.loss(pred_loss, pp_target)
        pred_log = self.postprocess_pred_logging(pred)
        self.logger.log(mode, epoch, pred_log, target, loss.item())

    def reset(self):
        self.logger.reset()

    def validation_epoch_finish(self, epoch):
        self.early_stopper.early_stop(self.logger.get_score("val", "loss"))

    def training_finish(self):
        self.model.load_state_dict(self.early_stopper.best_state_dict)

    @torch.no_grad()
    def test_batch_time(self, x, target, epoch, mode):
        self.model.eval()

        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t2 = torch.cuda.Event(enable_timing=True)
        t3 = torch.cuda.Event(enable_timing=True)
        t4 = torch.cuda.Event(enable_timing=True)
        t5 = torch.cuda.Event(enable_timing=True)
        t6 = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        t0.record()
        x = self.preprocess_input(x)  # Step 0
        torch.cuda.synchronize()
        t1.record()
        pred = self.model(x)  # Step 1
        torch.cuda.synchronize()
        t2.record()
        pred_loss = self.postprocess_pred_loss(pred)  # Step 2
        torch.cuda.synchronize()
        t3.record()
        pp_target = self.preprocess_target(target)  # Step 3
        torch.cuda.synchronize()
        t4.record()
        loss = self.loss(pred_loss, pp_target)  # Step 4
        torch.cuda.synchronize()
        t5.record()
        _ = self.postprocess_pred_logging(pred)  # Step 5
        torch.cuda.synchronize()
        t6.record()

        return [
            t0.elapsed_time(t1),
            t1.elapsed_time(t2),
            t2.elapsed_time(t3),
            t3.elapsed_time(t4),
            t4.elapsed_time(t5),
            t5.elapsed_time(t6),
        ], loss
