import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from inspect_raw_wm811k import (
    DATA_FILE,
    load_legacy_pandas_pickle,
    normalize_nested_value,
)
from wafermap.constants import RANDOM_SEED, WM811K_CLASS_NAMES
from wafermap.paths import RESULT_DIR

OUTPUT_DIR = RESULT_DIR / "figures" / "data_overview"

WAFER_COLORMAP = ListedColormap(
    ["#FFFFFF", "#B8C2CC", "#D62728"]
)

def load_labeled_data():
    print(f"正在读取原始数据：{DATA_FILE}")
    data = load_legacy_pandas_pickle(DATA_FILE)
    data["label"] = data["failureType"].map(normalize_nested_value)
    labeled_data = data[data["label"].isin(WM811K_CLASS_NAMES)].copy()
    print(f"有标签样本数量：{len(labeled_data):,}")
    return labeled_data


def plot_class_examples(labeled_data):
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))

    for axis, class_name in zip(axes.flat, WM811K_CLASS_NAMES):
        class_rows = labeled_data[labeled_data["label"] == class_name]
        example = class_rows.sample(n=1, random_state=RANDOM_SEED).iloc[0]
        wafer_map = example["waferMap"]

        axis.imshow(
            wafer_map,
            cmap=WAFER_COLORMAP,
            vmin=0,
            vmax=2,
            interpolation="nearest",
        )
        axis.set_title(
            f"{class_name}\n"
            f"Size: {wafer_map.shape[0]} x {wafer_map.shape[1]}"
        )
        axis.axis("off")

    fig.suptitle("WM-811K: One Example per Class", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    output_file = OUTPUT_DIR / "wm811k_class_examples.png"
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"九类示例图已保存：{output_file}")

def plot_class_distribution(labeled_data):
    class_counts = (
        labeled_data["label"]
        .value_counts()
        .reindex(WM811K_CLASS_NAMES)
    )

    bar_colors = [
        "#D55E00" if class_name != "none" else "#4C78A8"
        for class_name in class_counts.index
    ]

    fig, axis = plt.subplots(figsize=(11, 6))
    bars = axis.bar(
        class_counts.index,
        class_counts.values,
        color=bar_colors,
    )

    axis.set_yscale("log")
    axis.set_title("WM-811K Class Distribution")
    axis.set_xlabel("Failure Class")
    axis.set_ylabel("Number of Wafers (Log Scale)")
    axis.grid(axis="y", which="both", linestyle="--", alpha=0.35)
    axis.set_axisbelow(True)

    axis.bar_label(
        bars,
        labels=[f"{int(count):,}" for count in class_counts.values],
        padding=3,
        fontsize=8,
    )
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right")

    fig.tight_layout()
    output_file = OUTPUT_DIR / "wm811k_class_distribution.png"
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"类别分布图已保存：{output_file}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    labeled_data = load_labeled_data()
    plot_class_examples(labeled_data)
    plot_class_distribution(labeled_data)

    print("WM-811K 数据可视化完成。")


if __name__ == "__main__":
    main()