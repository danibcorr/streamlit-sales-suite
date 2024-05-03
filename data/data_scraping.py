# %% Libraries

# Import the downloader function from the bing_image_downloader library
from bing_image_downloader import downloader

# %% Functions

def download_images(query_search: str, num_img_download: int, output_path: str) -> None:
    
    """
    Download images from Bing using the given query search and save them to the specified output path.

    Args:
        query_search (str): The search query to use for downloading images.
        num_img_download (int): The number of images to download.
        output_path (str): The path to save the downloaded images.
    """
    
    try:

        # Download images using the downloader function
        downloader.download(query_search, limit=num_img_download, output_dir=output_path, adult_filter_off=True, force_replace=False, timeout=60, verbose=True)
    
    except Exception as e:

        # Print an error message if an exception occurs
        print(f"Error downloading images: {e}")

def main(dataset_path: str) -> None:

    """
    Main function to download images from Bing.
    """
    
    # Get the search query from the user
    query_search = input("Enter the search query: ")
    
    # Get the number of images to download from the user
    num_img_download = int(input("Enter number of images to download: "))
    
    # Get the folder name from the user
    folder_name = input("Enter the folder name: ")
    
    # Construct the output path by combining the dataset path and folder name
    output_path = dataset_path + folder_name
    
    # Call the download_images function to download images
    download_images(query_search, num_img_download, output_path)

# %% Main

if __name__ == '__main__':
    
    dataset_path = './datasets/items/'
    main(dataset_path)