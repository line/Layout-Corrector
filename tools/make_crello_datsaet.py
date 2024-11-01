import argparse
import os

import torch
from torch_geometric.data import Data
from torch_geometric.data.collate import collate as _collate
from datasets import list_datasets, load_dataset, list_metrics, load_metric
from torch import Tensor
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple


def collate(data_list: List[Data]) -> Tuple[Data, Optional[Dict[str, Tensor]]]:
    r"""Collates a Python list of :obj:`torch_geometric.data.Data` objects
    to the internal storage format of
    :class:`~torch_geometric.data.InMemoryDataset`."""
    if len(data_list) == 1:
        return data_list[0], None

    data, slices, _ = _collate(
        data_list[0].__class__,
        data_list=data_list,
        increment=False,
        add_batch=False,
    )

    return data, slices


label_names = [
    'coloredBackground', 'imageElement', 'maskElement', 'svgElement', 'textElement'
]

def save_dataset(save_dir, split='train', max_seq_length=25):

    dataset = load_dataset('cyberagent/crello')
    assert split in dataset.keys()
    dataset = dataset[split]
    features = dataset.features

    skip_count = 0
    data_list = []

    for example in tqdm(dataset):
        canvas_width = int(features["canvas_width"].int2str(example["canvas_width"]))
        canvas_height = int(features["canvas_height"].int2str(example["canvas_height"]))

        N = example['length']
        if N == 0 or max_seq_length < N:
            skip_count += 1
            continue

        boxes = []
        labels = []

        for ele_id in range(N):
            # bbox
            left, top = example['left'][ele_id], example['top'][ele_id]
            width, height = example['width'][ele_id], example['height'][ele_id]
            label = example['type'][ele_id]
            cx = (left + (left + width)) / 2.
            cy = (top + (top + height)) / 2.
            b = [cx, cy, width, height]
            boxes.append(b)
            labels.append(label)

        boxes = torch.tensor(boxes, dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.long)

        data = Data(x=boxes, y=labels)
        data.attr = {
            "name": example['id'],
            "width": canvas_width,
            "height": canvas_height,
            "filtered": False,
            "has_canvas_element": False,
            "NoiseAdded": False,
        }
        data_list.append(data)

    filename = dict(
        train='train.pt',
        validation='val.pt',
        test='test.pt',
    )
    save_path = os.path.join(save_dir, filename[split])

    with open(save_path, "wb") as file_obj:
        torch.save(collate(data_list), file_obj)

    print(len(data_list), 'samples')
    print('skiped: ', skip_count)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('save_dir')
    parser.add_argument('--max_seq_length', type=int, default=25)
    args = parser.parse_args()

    max_seq_length = args.max_seq_length
    save_dir = os.path.join(args.save_dir, f'crello-bbox-max{max_seq_length}')

    os.makedirs(save_dir, exist_ok=True)
    save_dataset(save_dir, split='train', max_seq_length=max_seq_length)
    save_dataset(save_dir, split='validation', max_seq_length=max_seq_length)
    save_dataset(save_dir, split='test', max_seq_length=max_seq_length)