import torch


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")
        self.early_stopped = False

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
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
            self.opt = torch.optim.SGD(self.model.parameters(), lr=lr)
        elif optimizer == "Adam":
            self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer == "AdamW":
            self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr)

        self.reset()

        self.nr_training_steps = 0
        self.nr_test_steps = 0

        self.early_stopper = EarlyStopper(patience=10, min_delta=0)

    def train_batch(self, x, target, epoch):
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
            self.logger.log("train", epoch, pred_log, target)

            self.nr_training_steps += 1
            if self.verbose:
                if self.nr_training_steps % 100 == 0:
                    print(f"Step {self.nr_training_steps}: ", loss.item())

        return loss

    @torch.no_grad()
    def test_batch(self, x, target, epoch, mode):
        self.model.eval()
        x = self.preprocess_input(x)
        pred = self.model(x)
        pred_log = self.postprocess_pred_logging(pred)
        self.logger.log(mode, epoch, pred_log, target)

    def reset(self):
        self.logger.reset()