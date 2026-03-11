"""
Download and setup emotion2vec pretrained model for fine-tuning
Model: https://huggingface.co/emotion2vec/emotion2vec_base
"""

import os
from huggingface_hub import snapshot_download

def download_emotion2vec():
    """Download emotion2vec_base model from Hugging Face"""
    
    print("=" * 70)
    print("DOWNLOADING emotion2vec_base MODEL")
    print("=" * 70)
    print()
    
    # Set download directory
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'emotion2vec', 'emotion2vec_base')
    os.makedirs(os.path.dirname(model_dir), exist_ok=True)
    
    print(f"Download directory: {model_dir}")
    print()
    
    try:
        print("Downloading model files from Hugging Face...")
        print("This may take several minutes depending on your connection...")
        print()
        
        # Download model
        downloaded_path = snapshot_download(
            repo_id="emotion2vec/emotion2vec_base",
            local_dir=model_dir,
            local_dir_use_symlinks=False
        )
        
        print()
        print("=" * 70)
        print("DOWNLOAD COMPLETE!")
        print("=" * 70)
        print(f"Model saved to: {downloaded_path}")
        print()
        
        # List downloaded files
        print("Downloaded files:")
        for root, dirs, files in os.walk(model_dir):
            for file in files:
                file_path = os.path.join(root, file)
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                rel_path = os.path.relpath(file_path, model_dir)
                print(f"  {rel_path} ({size_mb:.2f} MB)")
        
        print()
        print("Next steps:")
        print("  1. Run fine-tuning script on English dataset")
        print("  2. Run fine-tuning script on Tamil dataset")
        print("  3. Compare results with CNN models")
        
        return downloaded_path
        
    except Exception as e:
        print(f"Error downloading model: {str(e)}")
        print()
        print("Alternative: You can manually download from:")
        print("https://huggingface.co/emotion2vec/emotion2vec_base/tree/main")
        return None

if __name__ == '__main__':
    download_emotion2vec()
