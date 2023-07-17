@hydra.main(version_base=None, config_path="../yamls", config_name="config")
def main(config: DictConfig) -> None:
    logging.debug("./logs/log_{0:%Y%m%d%H%M%S}.log".format(now))

    logging.debug("config: {}".format(options.config))
    logging.debug(feats)
    logging.debug(params)

    # 指定した特徴量からデータをロード
    X_train_all, X_test = load_datasets(feats)


if __name__ == "__main__":
    main()
