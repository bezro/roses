import os
from multiprocessing import Process

import fire
import mlflow
import numpy as np
import onnxruntime
from roses_dataset import RosesDataset, load_roses_dataset


def run_server(model, up=False, server="http://localhost:5000"):
    assert model in ["small", "medium", "large", "xlarge"]
    port = server.split(":")[-1]

    def up_server():
        os.system(f"mlflow ui -p {port}")

    if up:
        Process(target=up_server, daemon=True).start()
    mlflow.set_tracking_uri(server)
    experiment = mlflow.set_experiment("inference")

    X, y = load_roses_dataset("myremote")
    dataset = RosesDataset(X, y, slice(None))

    onnx_session = onnxruntime.InferenceSession(
        f"./model/net_{model}.onnx", providers=["CPUExecutionProvider"]
    )
    input_name = onnx_session.get_inputs()[0].name

    true = []
    pred = []

    with mlflow.start_run(
        run_name=f"inference_{model}", experiment_id=experiment.experiment_id
    ):
        mlflow.log_param("model", model)
        with open(f"./data/results_{model}.csv", "w") as file:
            file.write("x,y(true),y(pred)\n")
            for x, y in dataset:
                a = onnx_session.run(None, {input_name: x.reshape(1, -1)})[0]
                true.append(y)
                pred.append(a)
                file.write("{},{},{}\n".format(x, y, a))

        pred = (np.hstack(pred) > 0.5).astype("float32")
        true = (np.hstack(true) > 0.5).astype("float32")

        mlflow.log_metric("accuracy", np.mean(pred == true))
        mlflow.log_metric("precision", np.sum(pred * true) / np.sum(true))
        mlflow.log_metric("recall", np.sum(pred * true) / np.sum(pred))


if __name__ == "__main__":
    fire.Fire(run_server)
