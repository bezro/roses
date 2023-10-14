import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../config", config_name="rfc.yaml", version_base="1.3")
def train(cfg: DictConfig):
    from pickle import dump

    from sklearn.datasets import load_breast_cancer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    rfc = RandomForestClassifier(
        n_estimators=cfg.rfc.n_estimators, max_depth=cfg.rfc.max_depth
    )
    rfc.fit(X_train, y_train)

    with open("./model/rfc.pkl", "wb") as file:
        dump(rfc, file)

    y_pred = rfc.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("accuracy on test data:", accuracy)


if __name__ == "__main__":
    train()
