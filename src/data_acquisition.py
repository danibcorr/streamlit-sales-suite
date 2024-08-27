from loguru import logger

from data.data_cleaning import data_cleaning
from data.data_scraping import data_scraping
from config import config as cg

# Log storage file
logger.add(cg.LOGS_DATA_ACQUISITION, rotation="1 MB", compression="zip")


def main(dataset_path: str) -> None:

    # First we obtain the data by web scraping.
    logger.info("Escrapping data from Bing...")
    data_scraping(dataset_path)

    # Then we perform data cleaning
    logger.info("Data cleaning...")
    data_cleaning(dataset_path)


if __name__ == "__main__":

    main(cg.IMGS_DATASET_PATH)
