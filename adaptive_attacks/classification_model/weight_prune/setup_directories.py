"""
Setup Directory Structure

This script creates the necessary directory structure for the project.
Run this before training to ensure all directories exist.
"""

import os
import config as cfg


def create_directory_structure():
    """
    Create all necessary directories for the project.
    """
    directories = [
        cfg.INIT_MODEL_DIR,
        cfg.PARENT_MODEL_DIR,
        cfg.CHILD_MODEL_DIR,
        './logs',
        './results',
        './checkpoints'
    ]
    
    print("Creating directory structure...")
    print("=" * 60)
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✓ Created: {directory}")
        else:
            print(f"○ Exists: {directory}")
    
    print("=" * 60)
    print("Directory structure setup complete!\n")


def verify_dataset():
    """
    Verify that the dataset path exists and is accessible.
    """
    print("Verifying dataset...")
    print("=" * 60)
    
    dataset_path = cfg.DATA_PATH
    data_folder = os.path.join(dataset_path, '101_ObjectCategories')
    
    if os.path.exists(data_folder):
        # Count number of categories
        categories = [d for d in os.listdir(data_folder) 
                     if os.path.isdir(os.path.join(data_folder, d))]
        num_categories = len(categories)
        
        print(f"✓ Dataset found at: {dataset_path}")
        print(f"✓ Number of categories: {num_categories}")
        
        if num_categories < 101:
            print(f"⚠ Warning: Expected 101 categories, found {num_categories}")
    else:
        print(f"✗ Dataset not found at: {dataset_path}")
        print(f"\nPlease download Caltech-101 and update DATA_PATH in config.py")
        print(f"Download from: http://www.vision.caltech.edu/Image_Datasets/Caltech101/")
    
    print("=" * 60 + "\n")


def print_project_structure():
    """
    Print the expected project structure.
    """
    structure = """
Expected Project Structure:
===========================

Small_Model/
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── config.py                # Configuration file
├── train_caltech_new.py     # Main training script
├── caltech101_new.py        # Dataset loader
├── utils.py                 # Helper functions
├── evaluate.py              # Evaluation script
├── setup_directories.py     # This script
│
├── data/                    # Dataset directory
│   └── caltech-101/
│       └── 101_ObjectCategories/
│
├── initmodels/              # Initial model states
├── Pmodels/                 # Parent model checkpoints
├── Cmodels/                 # Child model checkpoints
├── logs/                    # Training logs
├── results/                 # Evaluation results
└── checkpoints/             # Training checkpoints
    """
    print(structure)


def main():
    """Main setup function"""
    print("\n" + "=" * 60)
    print("HIERARCHICAL FINE-TUNING LINEAGE - PROJECT SETUP")
    print("=" * 60 + "\n")
    
    # Create directories
    create_directory_structure()
    
    # Verify dataset
    verify_dataset()
    
    # Print project structure
    print_project_structure()
    
    print("\nSetup complete! Next steps:")
    print("1. Ensure Caltech-101 dataset is downloaded and DATA_PATH is correct")
    print("2. Review and modify config.py as needed")
    print("3. Run: python train_caltech_new.py --mode train")
    print("4. Evaluate: python evaluate.py --model_type parent\n")


if __name__ == "__main__":
    main()
