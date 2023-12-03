from pickle import load

import fire
import onnxruntime
from roses_dataset import RosesDataset, load_roses_dataset
from torch.utils.data import DataLoader


def infer(format, model):
    assert format in ["pkl", "onnx"]
    assert model in ["small", "medium", "large", "xlarge"]
    X, y = load_roses_dataset("myremote")

    model_filename = f"./model/net_{model}.{format}"

    dataset = RosesDataset(X, y, slice(None))

    if format == "pkl":
        dataloader = DataLoader(dataset, batch_size=1)

        with open(model_filename, "rb") as file:
            net = load(file)

        with open(f"./data/results_{model}.csv", "w") as file:
            file.write("x,y\n")
            for x, y in dataloader:
                a = net(x).item()
                file.write("{},{},{}\n".format(x, y, a))
    else:
        onnx_session = onnxruntime.InferenceSession(
            model_filename, providers=["CPUExecutionProvider"]
        )

        input_name = onnx_session.get_inputs()[0].name

        with open(f"./data/results_{model}.csv", "w") as file:
            file.write("x,y(true),y(pred)\n")
            for x, y in dataset:
                a = onnx_session.run(None, {input_name: x.reshape(1, -1)})[0]
                file.write("{},{},{}\n".format(x, y, a))


if __name__ == "__main__":
    fire.Fire(infer)
