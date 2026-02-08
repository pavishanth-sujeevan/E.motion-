"""
Setup script to create the required directory structure
Run this first to set up your project
"""
import os


def create_directory_structure():
    """Create all required directories for the project"""

    # Define directory structure
    directories = [
        'data/raw/RAVDESS-SPEECH',
        'data/raw/TESS/TESS Toronto emotional speech set data',
        'data/processed',
        'models/saved_models',
        'logs/training_logs',
        'results',
        'src'
    ]

    print("=" * 60)
    print("CREATING PROJECT DIRECTORY STRUCTURE")
    print("=" * 60)

    created = []
    already_exists = []

    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            created.append(directory)
            print(f"✓ Created: {directory}")
        else:
            already_exists.append(directory)
            print(f"→ Already exists: {directory}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Directories created: {len(created)}")
    print(f"Already existed: {len(already_exists)}")

    if created:
        print("\n✓ Project structure created successfully!")
    else:
        print("\n✓ All directories already exist!")

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("\n1. Place your datasets in the following directories:")
    print("   - RAVDESS: data/raw/RAVDESS-SPEECH/")
    print("   - TESS: data/raw/TESS/TESS Toronto emotional speech set data/")
    print("\n2. Run: python src/preprocess.py")
    print("3. Run: python src/train.py")
    print("4. Run: python src/evaluate.py")

    print("\nOr run: python demo.py to check the current status")
    print("=" * 60)


if __name__ == "__main__":
    create_directory_structure()