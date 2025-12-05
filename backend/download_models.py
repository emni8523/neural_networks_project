import os
import gdown

def download_models():
    """Download model folders if they don't exist"""
    
    # Download json_files folder
    if not os.path.exists('json_files'):
        print("Downloading json_files folder...")
        gdown.download_folder(
            id='1mUXb8vDy-I7mGC6LfIDhGNrdKXxuFZlk', 
            quiet=False,
            use_cookies=False
        )
    
    # Download Model folder
    if not os.path.exists('Model'):
        print("Downloading Model folder...")
        gdown.download_folder(
            id='1VHfUjjXKYA5I5Tkn8wclAu8F3-rr6mEW',  
            quiet=False,
            use_cookies=False
        )
    
    # Download single model file
    if not os.path.exists('keras_folder'):
        print("Downloading model2new.keras...")
        gdown.download(
            id='1t1U0SlrdnE1uUpQ96D7teTG-n361WGcT',  
            output='model2new.keras',
            quiet=False
        )
    
    print("All models downloaded successfully!")

if __name__ == "__main__":
    download_models()
