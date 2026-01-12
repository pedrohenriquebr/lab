import requests
import os
import tempfile



import requests
from tqdm import tqdm
import os

MINERL_DATASET_ZIP_URL='https://archive.org/download/minerl_navigate/minerl_navigate.zip'
DATASETS_DIR = os.path.join(os.getcwd(), 'datasets')



def download_file_with_progress(url, filename):
    """
    Downloads a file from a URL with a progress bar.
    """
    # Stream the download
    response = requests.get(url, stream=True)
    response.raise_for_status() # Raise an exception for bad status codes

    # Get the total file size from the 'Content-Length' header
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 1024 # 1 Kibibyte

    # Open the file in binary write mode and use tqdm for progress tracking
    with tqdm(total=total_size_in_bytes, unit="B", unit_scale=True, desc=filename) as progress_bar:
        with open(filename, "wb") as file:
            for chunk in response.iter_content(block_size):
                if chunk: # Filter out keep-alive chunks
                    file.write(chunk)
                    progress_bar.update(len(chunk))

    # Optional: Check if the download completed successfully
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR: Download progress did not match the total file size.")
    else:
        print(f"\nSuccessfully downloaded {filename}")



def main():
    temp_dir = tempfile.gettempdir()
    zip_temp = os.path.join(temp_dir, "minerl_navigate.zip")
    print(f"🌐 Usando diretório temporário: {temp_dir}")
    
    if os.path.exists(zip_temp) and os.path.getsize(zip_temp) > 199_000_000:
        print("📂 Arquivo já existe no diretório temporário, pulando download.")
    
    else:
        print("🚀 Baixando dataset MineRL Navigate (pode demorar)...")
        download_file_with_progress(MINERL_DATASET_ZIP_URL, zip_temp)
        print("✅ Download completo.")

    print('Criando diretório de datasets se não existir...')
    os.makedirs(DATASETS_DIR, exist_ok=True)
    print("📂 Extraindo dataset...")
    import zipfile
    with zipfile.ZipFile(os.path.join(temp_dir, "minerl_navigate.zip"), 'r') as zip_ref:
        zip_ref.extractall(DATASETS_DIR)
    
    print("📂 Renomeando pasta extraída...")
    os.rename(os.path.join(DATASETS_DIR, "minerl_navigate"), os.path.join(DATASETS_DIR, "minerl"))
    
    print("✅ Extração completa.")







    # Example Usage:
    # Replace with the URL of the file you want to download
    download_url = "http://www.ovh.net/files/10Mb.dat" 
    output_filename = "downloaded_file.dat"

if __name__ == "__main__":
    main()
