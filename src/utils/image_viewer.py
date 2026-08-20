import cv2
import matplotlib.pyplot as plt

class ImageViewer:
    def view(image, ann):   
        boxes = []


        img = cv2.imread(img)
        if img is None:
            raise FileNotFoundError("Image not found")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        with open(ann, "r") as f:
            for line in f:
                values = line.strip().split(",")

                x = int(values[0])
                y = int(values[1])
                w = int(values[2])
                h = int(values[3])
                cls = int(values[5])
                if cls == 0:
                    continue

                boxes.append((x, y, w, h, cls))

        plt.figure(figsize=(12,8))
        plt.imshow(img)

        ax = plt.gca()

        for x, y, w, h, cls in boxes:

            rect = plt.Rectangle(
                (x, y),
                w,
                h,
                linewidth=1.5,
                edgecolor="green",
                facecolor="none"
            )

            ax.add_patch(rect)

            plt.text(
                x,
                y,
                f"class {cls}",
                color="yellow",
                fontsize=8,
                bbox=dict(facecolor="black", alpha=0.5)
            )


        plt.axis("off")
        plt.show()