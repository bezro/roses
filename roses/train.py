import hydra
import mlflow
import numpy as np
import torch.onnx
from net_model import Net, accuracy, eval, train_epoch
from omegaconf import DictConfig
from roses_dataset import RosesDataset, load_roses_dataset
from torch.nn import BCELoss
from torch.optim import Adam
from torch.utils.data import DataLoader


@hydra.main(config_path="../config", config_name="rfc.yaml", version_base="1.3")
def train(cfg: DictConfig):
    from pickle import dump

    mlflow.set_tracking_uri(cfg.mlflow.server)
    mlflow.set_experiment(cfg.mlflow.experiment)

    X, y = load_roses_dataset("myremote")
    train_mask = np.random.random(size=len(y)) > cfg.test_size
    dataset_train = RosesDataset(X, y, train_mask)
    dataset_test = RosesDataset(X, y, ~train_mask)
    dataloader_train = DataLoader(dataset_train, batch_size=cfg.batch_size)
    dataloader_test = DataLoader(dataset_test, batch_size=cfg.batch_size)

    dataloader_onnx = DataLoader(dataset_train, batch_size=1)

    net_cfgs = [cfg.net.small, cfg.net.medium, cfg.net.large, cfg.net.xlarge]
    for net_cfg in net_cfgs:
        with mlflow.start_run(run_name=f"run_{net_cfg.name}"):
            mlflow.log_param(
                "layer_sizes",
                (net_cfg.hidden_layer1, net_cfg.hidden_layer2, net_cfg.hidden_layer3),
            )
            net = Net(
                net_cfg.hidden_layer1, net_cfg.hidden_layer2, net_cfg.hidden_layer3
            )
            opt = Adam(net.parameters(), lr=cfg.lr)

            print(f"model: {net_cfg.name}")
            for epoch in range(cfg.n_epochs):
                train_epoch(net, opt, BCELoss(reduction="mean"), dataloader_train)

                train_loss = eval(net, BCELoss(reduction="sum"), dataloader_train)
                test_loss = eval(net, BCELoss(reduction="sum"), dataloader_test)
                print(
                    "epoch: {:d}, train BCE: {:.3f}, test BCE: {:.3f}".format(
                        epoch, train_loss, test_loss
                    )
                )

                mlflow.log_metric("train loss", train_loss, step=epoch)
                mlflow.log_metric("test loss", test_loss, step=epoch)

                test_accuracy = accuracy(net, dataloader_test)
                mlflow.log_metric("test accuracy", test_accuracy, step=epoch)

            with open(f"./model/net_{net_cfg.name}.pkl", "wb") as file:
                dump(net, file)

            for x, _ in dataloader_onnx:
                torch.onnx.export(
                    net, x, f"./model/net_{net_cfg.name}.onnx", export_params=True
                )

            print()


if __name__ == "__main__":
    train()
