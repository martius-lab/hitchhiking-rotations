import torch


class Trainer:
    def __init__(
        self,
        preprocess_target,
        preprocess_input,
        postprocess_pred,
        postprocess_logging,
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
        self.postprocess_pred = postprocess_pred
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

    def train_batch(self, x, target):
        self.opt.zero_grad()

        with torch.no_grad():
            pp_target = self.preprocess_target(target)

        x = self.preprocess_input(x)
        pred = self.model(x)
        pred = self.postprocess_pred(pred)

        loss = self.loss(pred, pp_target)
        loss.backward()
        self.opt.step()

        with torch.no_grad():
            pred = self.postprocess_pred_logging(pred)
            self.logger.log("train", pred, target)

            self.nr_training_steps += 1
            if self.verbose:
                if self.nr_training_steps % 100 == 0:
                    print(f"Training step {self.nr_training_steps}: ", loss.item())

        return loss

    @torch.no_grad()
    def test_batch(self, x, target):
        target = self.preprocess_target(target)

        x = self.preprocess_input(x)
        pred = self.model(x)
        pred = self.postprocess_pred(pred)
        pred = self.postprocess_pred_logging(pred)
        self.logger.log("test", pred, target)

    def reset(self):
        self.logger.reset()
