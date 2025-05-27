import os
from typing import Any, Tuple, Optional, Callable

import pandas as pd
import PIL

from torchvision.datasets import VisionDataset


class CustomVisionDataset(VisionDataset):
    def __init__(
        self,
        *,
        root: str,
        labelColumn: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        preloading: bool = False,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)

        # Initialize the dataset with root directory and extra information
        self.root = root
        self.labelColumn = labelColumn

        # if joining multiple datasets, assume root and labelColumn as strings separated by ;
        all_roots = root.split(";")
        all_labelColumns = labelColumn.split(";")

        all_dfs = []
        for r, lc in zip(all_roots, all_labelColumns):
            # Load the dataset from the root directory
            df = pd.read_csv(os.path.join(r, "labels.csv"))
            # add column for full path to image: join root and image column
            df["full_image_path"] = df["image"].apply(
                lambda x: os.path.join(r, "images", x)
            )
            # add column for same consistent label column
            df["label"] = df[lc]
            # append to list
            all_dfs.append(df)

        # concatenate all dataframes
        self.data = pd.concat(all_dfs, ignore_index=True)
        
        self.image_paths = self.data["full_image_path"].tolist()
        self.labels = self.data["label"].tolist()

        self.images = [None] * len(self.image_paths)
        self.preloading = preloading
        if preloading:
            import tqdm
            # If preloading is enabled, we will load all images into memory, parallelized
            from concurrent.futures import ThreadPoolExecutor
            def load_image(path):
                return PIL.Image.open(path).convert("RGB")
            
            with ThreadPoolExecutor() as executor:
                self.images = list(tqdm.tqdm(executor.map(load_image, self.image_paths), total=len(self.image_paths), desc="Preloading images"))
        else:
            # If preloading is disabled, we will load images on-the-fly
            self.images = None



    def __getitem__(self, index):
        # Get the image path and label for the given index
        image_path = self.image_paths[index]
        label = self.labels[index]

        # Load the image from the path
        if self.preloading:
            # If preloading is enabled, return the preloaded image
            image = self.images[index]
        else:
            # Otherwise, load the image from disk
            image = PIL.Image.open(image_path).convert("RGB")

        if self.transforms is not None:
            image, label = self.transforms(image, label)

        return image, label
    
    def __len__(self) -> int:
        # Return the number of samples in the dataset
        return len(self.image_paths)



