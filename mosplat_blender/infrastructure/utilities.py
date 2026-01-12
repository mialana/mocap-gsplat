from dotenv import load_dotenv
from pathlib import Path


def load_dotenv_file():
    dotenv_path = Path(__file__).resolve().parent.parent.joinpath(".env")
    load_dotenv(
        dotenv_path, verbose=True, override=True
    )  # load all env variables from local `.env`
