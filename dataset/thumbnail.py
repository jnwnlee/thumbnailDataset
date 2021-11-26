from torchvision.datasets.vision import VisionDataset
from PIL import Image

import numpy as np
import os
import os.path
from typing import Any, Callable, cast, Dict, List, Optional, Tuple

import pandas as pd

def has_file_allowed_extension(filename: str, extensions: Tuple[str, ...]) -> bool:
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)

def default_loader(path: str) -> Any:
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

# revising torchvision.datasets.DatasetFolder 
class ThumbnailDataset(VisionDataset):
    def __init__(self, root, loader, extensions=None, transform=None,
                 target_transform=None, is_valid_file=None):
        super(ThumbnailDataset, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        classes = [0, 1]
        class_to_idx = {'2': 1, '1': 0, '0': 0}
        samples = self.make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        # TODO: 아래 source code 보면서 고치기
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)
        self.loader = loader
        self.extensions = extensions
        self.classes = classes
        self.class_to_idx = class_to_idx

        # edit
        self.samples = [s[0] for s in samples]
        self.targets = [s[1] for s in samples]
        self.tags = [s[2] for s in samples]

	# edit
    def remove_data(self, filename):
        idx = np.where(self.file_names == filename)[0]
        self.targets = np.delete(self.targets, idx)
        self.samples = np.delete(self.samples, idx)
        self.tags = np.delete(self.tags, idx)
	
    def __getitem__(self, index):
      # edit
      path = self.samples[index]
      target = self.targets[index]
      tag = self.tags[index]

      sample = self.loader(path)

      if self.transform is not None:
        sample = self.transform(sample)
      if self.target_transform is not None:
        target = self.target_transform(target)

      # edit
      return sample, target, tag

    def _find_tags(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __len__(self):
        return len(self.samples)

    def split_idx(self, ratio):
        if not (type(ratio) == list or type(ratio) == tuple):
            raise TypeError('ratio must be a list or tuple')
        # TODO: finish
        tags, _ = self._find_tags(self.root)

        tag_to_idx = {tag: [] for tag in tags}
        for idx, tag in enumerate(self.tags):
            tag_to_idx[tag].append(idx)
        
        dataset_idx = {key: [] for key in ['train', 'val', 'test']}
        for indices in tag_to_idx.values():
            dataset_idx['train'] += indices[ : np.floor(len(indices)*ratio[0])]
            dataset_idx['val'] += indices[np.floor(len(indices)*ratio[0]) : np.floor(len(indices)*(ratio[0]+ratio[1]))]
            dataset_idx['test'] += indices[np.floor(len(indices)*(ratio[0]+ratio[1])) : ]

        return dataset_idx
    
    @staticmethod
    def make_dataset(self,
        directory: str,
        csv_directory: str,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        """Generates a list of samples of a form (path_to_sample, class).

        This can be overridden to e.g. read files from a compressed zip file instead of from the disk.

        See :class:`DatasetFolder` for details.

        Note: The class_to_idx parameter is here optional and will use the logic of the ``find_classes`` function
        by default.

        Args:
            directory (str): root dataset directory, corresponding to ``self.root``.
            class_to_idx (Dict[str, int]): Dictionary mapping class name to class index.
            extensions (optional): A list of allowed extensions.
                Either extensions or is_valid_file should be passed. Defaults to None.
            is_valid_file (optional): A function that takes path of a file
                and checks if the file is a valid file
                (used to check of corrupt files) both extensions and
                is_valid_file should not be passed. Defaults to None.

        Raises:
            ValueError: In case ``class_to_idx`` is empty.
            ValueError: In case ``extensions`` and ``is_valid_file`` are None or both are not None.
            FileNotFoundError: In case no valid file was found for any class.

        Returns:
            List[Tuple[str, int]]: samples of a form (path_to_sample, class)
        """

        directory = os.path.expanduser(directory)
        csv_directory = os.path.expanduser(directory)

        df = pd.read_csv(csv_directory)
        df = df['id', 'tag', 'OR']

        if class_to_idx is None:
            raise ValueError("The class_to_idx parameter cannot be None.")
        elif not class_to_idx:
            raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

        both_none = extensions is None and is_valid_file is None
        both_something = extensions is not None and is_valid_file is not None
        if both_none or both_something:
            raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

        if extensions is not None:

            def is_valid_file(x: str) -> bool:
                return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))

        is_valid_file = cast(Callable[[str], bool], is_valid_file)

        instances = []
        available_tags = set()
        # for target_class in sorted(class_to_idx.keys()):
        #     class_index = class_to_idx[target_class]
        #     target_dir = os.path.join(directory, target_class)
        #     if not os.path.isdir(target_dir):
        #         continue
        #     for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
        #         for fname in sorted(fnames):
        #             if is_valid_file(fname):
        #                 path = os.path.join(root, fname)
        #                 item = path, class_index
        #                 instances.append(item)

        #                 if target_class not in available_classes:
        #                     available_classes.add(target_class)
        tags, tag_to_idx = self._find_tags(directory)
        for root, _, fnames in sorted(os.walk(directory, followlinks=True)):
            for fname in sorted(fnames):
                if is_valid_file(fname):
                    if not any(df['id'].isin(fname)):
                        raise FileNotFoundError(str(fname)+' not in csv list.')
                    else:
                        row = df.loc[df['id'] == fname]
                        path = os.path.join(root, fname)
                        item = path, class_to_idx[row['OR']], row['tag']
                        instances.append(item)

                        if row['tag'] not in available_tags:
                            available_tags.add(row['tag'])



        empty_tags = set(tag_to_idx.keys()) - available_tags
        if empty_tags:
            msg = f"Found no valid file for the classes {', '.join(sorted(empty_tags))}. "
            if extensions is not None:
                msg += f"Supported extensions are: {', '.join(extensions)}"
            raise FileNotFoundError(msg)

        return instances