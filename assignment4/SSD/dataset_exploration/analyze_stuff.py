from tops.config import instantiate, LazyConfig
from ssd import utils
from tqdm import tqdm
from collections import defaultdict


def get_config(config_path):
    cfg = LazyConfig.load(config_path)
    cfg.train.batch_size = 1
    return cfg


def get_dataloader(cfg, dataset_to_visualize):
    if dataset_to_visualize == "train":
        # Remove GroundTruthBoxesToAnchors transform
        cfg.data_train.dataset.transform.transforms = cfg.data_train.dataset.transform.transforms[:-1]
        data_loader = instantiate(cfg.data_train.dataloader)
    else:
        cfg.data_val.dataloader.collate_fn = utils.batch_collate
        data_loader = instantiate(cfg.data_val.dataloader)

    return data_loader


def analyze_something(dataloader, cfg):
    label_counts = defaultdict(int)
    label_aspect_ratios = defaultdict(list)
    label_areas = defaultdict(list)
    image_widths = []
    image_heights = []

    for batch in tqdm(dataloader):
        for idx, label in enumerate(batch["labels"][0]):
            label_counts[label] += 1

            _, _, width, height = batch["boxes"][0][idx]
            label_aspect_ratios[label].append(width / height)
            label_areas[label].append(width * height)

        image_widths.append(batch["width"][0])
        image_heights.append(batch["height"][0])

    return {
        "label_counts": label_counts,
        "label_aspect_ratios": label_aspect_ratios,
        "label_areas": label_areas,
        "image_widths": image_widths,
        "image_heights": image_heights
    }


def main():
    config_path = "configs/tdt4265.py"
    cfg = get_config(config_path)
    dataset_to_analyze = "train"  # or "val"

    print("Label map is:", cfg.label_map)

    dataloader = get_dataloader(cfg, dataset_to_analyze)
    analyze_something(dataloader, cfg)


if __name__ == '__main__':
    main()
