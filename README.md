В данном репозитории приведено решение задачи бинарной классификации
предсказания наличия рака кожи.

Используется стандартный sklearn датасет breast cancer. Алгоритмом машинного
обучения является FeedForward Neural Network с тремя скрытыми слоями.

## Использование

### train:

`mlflow ui` `python3 roses/train.py`

### infer:

`python3 roses/infer.py --format="{onnx/pkl}" --model="{small/medium/large/xlarge}"`
`python3 roses/run_server.py --model="{small/medium/large/xlarge}" --up={True/False} --server=...`
up - поднимать ли новый сервер server - ip+порт сервера
