WM811K_CLASS_NAMES = (
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Near-full",
    "Random",
    "Scratch",
    "none",
)

NUM_CLASSES = len(WM811K_CLASS_NAMES)
IMAGE_SIZE = 64
RANDOM_SEED = 42

if __name__ == "__main__":
    print(f"Number of classes: {NUM_CLASSES}")
    print(f"Input image size: {IMAGE_SIZE} x {IMAGE_SIZE}")
    print("Class names:")

    for class_index, class_name in enumerate(WM811K_CLASS_NAMES):
        print(f"{class_index}: {class_name}")