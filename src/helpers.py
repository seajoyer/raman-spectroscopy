#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np

def create_metadata_dataframe(root_dir='../data'):
    """
    Scans a directory structure for text files and extracts metadata into a DataFrame.

    Args:
        root_dir (str): The root directory containing class folders (control, endo, exo).

    Returns:
        pd.DataFrame: A DataFrame containing file paths, labels, and extracted metadata.
    """
    data_info = []
    
    # Define class directories and their corresponding labels
    class_map = {'control': 0, 'endo': 1, 'exo': 2}

    for class_dir, label in class_map.items():
        class_path = os.path.join(root_dir, class_dir)
        
        # Skip if the class directory doesn't exist
        if not os.path.exists(class_path):
            continue

        for subdir in os.listdir(class_path):
            subpath = os.path.join(class_path, subdir)
            
            # Ensure we are iterating over directories
            if not os.path.isdir(subpath):
                continue

            for file in os.listdir(subpath):
                if file.endswith('.txt'):
                    full_path = os.path.join(subpath, file)
                    parts = file.split('_')
                    
                    # Parse filename parts
                    region = parts[0]
                    
                    # Safely extract 'center' and 'place' using list comprehensions
                    center_matches = [p for p in parts if 'center' in p]
                    place_matches = [p for p in parts if 'place' in p]

                    # Only append if the expected parts exist in the filename
                    if center_matches and place_matches:
                        center = center_matches[0].replace('center', '')
                        place = place_matches[0].replace('place', '')

                        data_info.append({
                            'path': full_path, 
                            'label': label, 
                            'subdir': subdir,
                            'region': region, 
                            'center': center, 
                            'place': place
                        })

    return pd.DataFrame(data_info)
