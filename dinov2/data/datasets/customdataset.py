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
        cfg: Optional[Any] = None
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)

        # Initialize the dataset with root directory and extra information
        self.root = root
        self.labelColumn = labelColumn
        if cfg is None:
            self.cfg = None
            self.make_patches = False
            self.patch_size = None
            self.patch_resize = None
            self.n_patches_per_axis = 1
        else:
            self.cfg = cfg
            self.make_patches = cfg.images.make_patches
            self.patch_size = cfg.images.patch_size
            self.patch_resize = cfg.images.patch_resize
            self.n_patches_per_axis = cfg.images.n_patches_per_axis



        # if joining multiple datasets, assume root and labelColumn as strings separated by ;
        all_roots = root.split(";")
        all_labelColumns = labelColumn.split(";")

        all_dfs = []
        for r, lc in zip(all_roots, all_labelColumns):
            if labelColumn is None or labelColumn == "":
                df = pd.DataFrame()
                # Just read images subdirectory
                images_path = os.path.join(r, "images")
                images_names = os.listdir(images_path)
                full_image_paths = [os.path.join(images_path, img) for img in images_names]
                df["full_image_path"] = full_image_paths
                df["label"] = 0 # Dummy label
            else:
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

        # drop all rows where label is -1
        self.data = self.data[self.data["label"] != -1].reset_index(drop=True)
        
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

    def _get_patch(self, image: Any, rel_index: int) -> Any:

        # Get image shape
        image_width, image_height = image.size

        stride_x = (image_width - int(self.patch_size * image_width)) // (self.n_patches_per_axis - 1)
        stride_y = (image_height - int(self.patch_size * image_height)) // (self.n_patches_per_axis - 1)

        # Get patch coordinates
        x = rel_index % self.n_patches_per_axis
        y = rel_index // self.n_patches_per_axis

        # Get patch
        patch = image.crop((x * stride_x, y * stride_y, x * stride_x + self.patch_size * image_width, y * stride_y + self.patch_size * image_height))

        patch = patch.resize((self.patch_resize, self.patch_resize))

        return patch

    def __getitem__(self, index):

        img_index = index // (self.n_patches_per_axis ** 2) if self.make_patches else index
        rel_index = index % (self.n_patches_per_axis ** 2) if self.make_patches else 0

        # Get the image path and label for the given index
        image_path = self.image_paths[img_index]
        label = self.labels[img_index]

        # Load the image from the path
        if self.preloading:
            # If preloading is enabled, return the preloaded image
            image = self.images[index]
        else:
            # Otherwise, load the image from disk
            image = PIL.Image.open(image_path).convert("RGB")

        if self.make_patches:
            image = self._get_patch(image, rel_index)

        if self.transforms is not None:
            image, label = self.transforms(image, label)

        return image, label
    
    def __len__(self) -> int:
        # Return the number of samples in the dataset
        return len(self.image_paths)



