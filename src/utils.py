import yaml
import logging

# Set logging
logging.basicConfig(level=logging.INFO, force=True)


# Loading config file helper function
def load_yaml_config(path: str):
    """Load yaml config file

    Args:
        path (str): Path to yaml config path

    Returns:
        dict: Config in dictionary format
    """
    logging.info(f"Loading the config file")
    with open(path) as stream:
        try:
            cfg = yaml.safe_load(stream)

            logging.info(f"Config file loaded successfully: {cfg}")
        except yaml.YAMLError as e:
            logging.info(f"{e}")

    return cfg


if __name__ == "__main__":
    pass
