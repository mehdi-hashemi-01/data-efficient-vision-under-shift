from figure_assets import copy_examples
from PIL import Image, ImageDraw
from cifar10_data import PROJECT, load_data


def show_examples():
    train, test = load_data()
    print("Train:", len(train), "| Test:", len(test))
    sheet = Image.new("RGB", (600, 240), "white")
    draw = ImageDraw.Draw(sheet)
    for label, name in enumerate(train.classes):
        index = train.targets.index(label)
        image, _ = train[index]
        x = (label % 5) * 120
        y = (label // 5) * 120
        sheet.paste(image.resize((96, 96)), (x, y))
        draw.text((x, y + 97), f"{label} {name}", fill="black")
        print(label, name, "| train:", train.targets.count(label), "| test:", test.targets.count(label))
    path = PROJECT / "cifar10_examples.png"
    sheet.save(path)
    print("Image sheet:", path)
    copy_examples(__file__, path)
    sheet.show()


if __name__ == "__main__":
    show_examples()
