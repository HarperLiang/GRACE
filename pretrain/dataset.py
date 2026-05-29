# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
import os
import random
import io
from typing import Any
import h5py
from typing import Callable, Optional, Tuple
from PIL import Image
import json
from torchvision.datasets import VisionDataset


class Decoder:
    def decode(self) -> Any:
        raise NotImplementedError


class ImageDataDecoder(Decoder):
    def __init__(self, image_data: bytes) -> None:
        self._image_data = image_data

    def decode(self) -> Image:
        f = io.BytesIO(self._image_data)
        return Image.open(f).convert(mode="RGB")


class TargetDecoder(Decoder):
    def __init__(self, target: Any):
        self._target = target

    def decode(self) -> Any:
        return self._target


class ExtendedVisionDataset(VisionDataset):
    def __init__(self, root: str, transforms: Optional[Callable] = None, transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None) -> None:
        super().__init__(root, transforms, transform, target_transform)

    def get_image_data(self, index: int) -> bytes:
        raise NotImplementedError

    def get_target(self, index: int) -> Any:
        raise NotImplementedError

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        try:
            image_data = self.get_image_data(index)
            image = ImageDataDecoder(image_data).decode()
        except Exception as e:
            raise RuntimeError(f"Cannot read image for sample {index}") from e

        target = self.get_target(index)
        target = TargetDecoder(target).decode()

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        raise NotImplementedError


class GC_FoundationDataset(ExtendedVisionDataset):
    """
    A dataset for reading JPEG-compressed patches from HDF5 files,
    each containing a 'patches' dataset with variable-length bytes.
    Samples up to `max_per_file` patches per file.
    """

    def __init__(self, root: str, transforms: Optional[Callable] = None, transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None, max_per_file: int = 5000) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.h5_files: List[str] = []
        self.index_map: List[Tuple[int, int]] = []

        random.seed(42)  # For reproducibility

        

        for file_idx, h5_path in enumerate(sorted([
            os.path.join(dp, f) for dp, dn, filenames in os.walk(root)
            for f in filenames if f.endswith('.h5')
        ])):
            try:
                with h5py.File(h5_path, 'r') as f:
                    if 'patches' not in f:
                        print(f"[Warning] '{h5_path}' does not contain 'patches'. Skipping.")
                        continue

                    num_patches = len(f['patches'])
                    if num_patches > max_per_file:
                        indices = random.sample(range(num_patches), max_per_file)
                    else:
                        indices = list(range(num_patches))

                    self.index_map.extend([(len(self.h5_files), i) for i in indices])
                    self.h5_files.append(h5_path)

            except OSError as e:
                print(f"[Warning] Skipping unreadable file '{h5_path}': {e}")

    def __len__(self) -> int:
        return len(self.index_map)

    def get_image_data(self, index: int) -> bytes:
        file_idx, patch_idx = self.index_map[index]
        h5_path = self.h5_files[file_idx]

        try:
            with h5py.File(h5_path, 'r') as f:
                byte_array = f['patches'][patch_idx].tobytes()
        except Exception as e:
            print(f"[Error] Failed to load patch {patch_idx} from {h5_path}: {e}")
            # Return a blank white patch
            img = Image.new("RGB", (224, 224), (255, 255, 255))
            with io.BytesIO() as buffer:
                img.save(buffer, format='JPEG')
                byte_array = buffer.getvalue()

        return byte_array

    def get_target(self, index: int) -> Any:
        # Dummy target for pretraining
        return 0
