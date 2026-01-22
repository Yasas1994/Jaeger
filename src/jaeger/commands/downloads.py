# curl -X GET  "https://ckan.fdm.uni-greifswald.de/api/3/action/package_search?q=jaeger_models"
import requests
from pathlib import Path
import tarfile
import logging

API_URL = "https://ckan.fdm.uni-greifswald.de/api/3/action/package_search"
QUERY = "jaeger"
logger = logging.getLogger("Jaeger")


def list_ckan_model_download_links(api_url: str = API_URL, query: str = QUERY):
    """
    Query the CKAN API to retrieve all resources associated with the specified dataset query,
    and extract direct download URLs for each available model.

    Parameters:
        api_url (str): The CKAN API endpoint for package search.
        query (str): The dataset search query.

    Returns:
        dict: A dictionary where keys are model names and values are their corresponding download URLs.
    """
    try:
        response = requests.get(api_url, params={"q": query})
        response.raise_for_status()
        result = response.json()

        if not result.get("success", False):
            raise ValueError("CKAN API returned an unsuccessful response.")

        datasets = result["result"]["results"]
        if not datasets:
            raise ValueError("No datasets found for the given query.")

        download_links = {}
        for dataset in datasets:
            for resource in dataset.get("resources", []):
                name = resource.get("name", resource.get("id"))
                url = resource.get("url")
                if url:
                    download_links[name] = url

        return download_links

    except Exception as e:
        print(f"Error retrieving model links: {e}")
        return {}


def download_file(download_link: tuple, output_dir: str):
    """
    Download files from given URLs and save them to the specified directory using pathlib.

    Parameters:
        download_links (dict): A dictionary of {filename: url} pairs.
        output_dir (str or Path): Directory where files will be saved.

    Returns:
        None
    """
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    name, url = download_link
    try:
        print(f"Downloading: {name} from {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Determine filename and preserve extension if not included
        file_path = output_path / name
        file_path = file_path.with_name(file_path.name + ".tar.gz")

        with file_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        logger.info(f"Saved to: {file_path}")
        extract_tar_archive(file_path)

    except Exception as e:
        print(e)
        logger.info(f"Failed to download {name} from {url}: {e}")


def extract_tar_archive(tar_path):
    """
    Extracts a .tar[.gz] file into the same directory as the archive,
    without creating a subdirectory. Removes the archive after extraction.

    Parameters:
        tar_path (str or Path): Path to the .tar or .tar.gz file.

    Returns:
        Path: Path to the directory where contents were extracted.
    """
    tar_path = Path(tar_path)
    extract_to = tar_path.parent  # Extract directly into current dir

    try:
        with tarfile.open(tar_path, mode="r:*") as tar:
            tar.extractall(path=extract_to)
        print(f"Extracted '{tar_path.name}' to '{extract_to}'")

        tar_path.unlink()  # Remove the archive
        print(f"Removed archive: {tar_path}")

        return extract_to

    except Exception as e:
        print(f"Failed to extract {tar_path}: {e}")
        return None


if __name__ == "__main__":
    model_links = list_ckan_model_download_links(API_URL, QUERY)
    print(model_links)
    download_file(
        ("jaeger_57341_1.5M", model_links["jaeger_57341_1.5M"]),
        output_dir="/home/yasas-wijesekara/Downloads/models",
    )
