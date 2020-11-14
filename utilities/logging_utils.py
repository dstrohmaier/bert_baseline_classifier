import logging


def log_dict(dict_name: str, dict_to_log: dict) -> None:
    logging.info(f"DICT: {dict_name}")

    for key, item in dict_to_log.items():
        logging.info(f"- {key}: {item}")
    logging.info("---End of dict---")