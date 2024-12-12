import cv2
import numpy as np
import os

class DisjointSet:
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def make_set(self, x):
        self.parent[x] = x
        self.rank[x] = 0

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x != root_y:
            if self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            elif self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            else:
                self.parent[root_y] = root_x
                self.rank[root_x] += 1


"""
TODO Two-pass algorithm
"""
def two_pass(img, connectivity):
    directions = [(-1, 0), (0, -1)]
    if connectivity == 8:
        directions.append((-1, -1))
        directions.append((-1, 1))

    label = np.full((img.shape[0], img.shape[1]), -1)

    ds = DisjointSet()

    #first pass    
    current = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j, 0] != 0 and img[i, j, 1] != 0 and img[i, j, 2] != 0:
                for dir in directions:
                    x = i + dir[0]
                    y = j + dir[1]
                    if x >= 0 and y >= 0 and x<img.shape[0] and y<img.shape[1] and img[x, y, 0] == img[i, j, 0] and img[x, y, 1] == img[i, j, 1] and img[x, y, 2] == img[i, j, 2]:
                        if label[i, j] == -1:
                            label[i, j] = label[x, y]
                        else:
                            ds.union(label[i, j], label[x, y])
                if label[i, j] == -1:
                    label[i, j] = current
                    ds.make_set(current)
                    current += 1

    #second pass
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            if label[i, j] != -1:
                label[i, j] = ds.find(label[i, j])


    label_count = {}

    # 統計每個 label 的 pixel 數量
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            if label[i, j] != -1:  # 忽略未標記的 pixel
                lbl = label[i, j]
                if lbl in label_count:
                    label_count[lbl] += 1
                else:
                    label_count[lbl] = 1

    current = {}
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            if label[i, j] != -1:  # 忽略未標記的 pixel
                lbl = label[i, j]
                if lbl in current:
                    current[lbl] += 1
                else:
                    current[lbl] = 1
                
                if current[lbl] != label_count[lbl] // 2 or label_count[lbl] < 200:
                    label[i, j] = -1

            
    return label

def main():
    os.makedirs("color_hints1", exist_ok=True)
    os.makedirs("color_hints2", exist_ok=True)
    for ind in range(2):
        if ind == 0:
            size = 6
        else:
            size = 10
        for index in range(size):
            seg = cv2.imread("segment{}/{}.jpg".format(ind+1, index+1))
            img = cv2.imread("post{}/{}_100.jpg".format(ind+1, index+1))
            
            label = two_pass(seg, 8)
            color_hint = img
            for i in range(label.shape[0]):
                for j in range(label.shape[1]):
                    if label[i, j] == -1:
                        color_hint[i, j] = (0, 0, 0)

            output_image = np.ones_like(img) * 0  # 全白背景

            # 定義圓形半徑
            radius = 5

            # 遍歷圖片中的每個像素
            height, width, _ = img.shape
            for y in range(height):
                for x in range(width):
                    pixel_color = img[y, x]
                    # 如果像素不是白色
                    if not np.array_equal(pixel_color, [0, 0, 0]):
                        # 在對應位置畫圓形
                        cv2.circle(output_image, (x, y), radius, pixel_color.tolist(), -1)

            cv2.imwrite("color_hints{}/color_hint{}_black.png".format(ind+1, index+1), output_image)
            #cv2.imshow("test", color_hint)
            #cv2.waitKey(0)
            #cv2.imshow("test", output_image)
            #cv2.waitKey(0)
        
if __name__ == "__main__":
    main()