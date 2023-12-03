from torch import nn


class Net(nn.Module):
    def __init__(self, hn1, hn2, hn3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LazyLinear(hn1),
            nn.ReLU(),
            nn.Linear(hn1, hn2),
            nn.ReLU(),
            nn.Linear(hn2, hn3),
            nn.ReLU(),
            nn.Linear(hn3, 1),
            nn.Sigmoid(),
            nn.Flatten(start_dim=0),
        )

    def forward(self, x):
        return self.net(x)


def train_epoch(
    model,
    opt,
    loss_fn,
    dataloader,
):
    model.train()
    for x, y in dataloader:
        opt.zero_grad()
        loss = loss_fn(model(x), y)

        loss.backward()
        opt.step()


def eval(model, loss_fn, dataloader):
    model.eval()

    loss_sum = 0.0
    loss_cnt = 0.0
    for x, y in dataloader:
        loss = loss_fn(model(x), y)

        loss_sum += loss.item()
        loss_cnt += len(y)

    return loss_sum / loss_cnt


def accuracy(model, dataloader):
    model.eval()

    metric_sum = 0.0
    metric_cnt = 0.0
    for x, y in dataloader:
        a = (model(x) > 0.5).float()

        metric_sum += (a == y).detach().numpy().sum()
        metric_cnt += len(y)

    return metric_sum / metric_cnt
