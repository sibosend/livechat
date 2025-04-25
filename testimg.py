import cv2
from datetime import datetime
            
import numpy as np
from PIL import Image
from rembg import remove

i0 = cv2.imread("/root/data/chazing_incubator/LiveTalking/data/avatars/avator_3/customvideo/image/00000000.png")
print(i0.shape)
i1 = cv2.imread("/root/data/chazing_incubator/LiveTalking/data/avatars/avator_3/00000000v1.png")
print(i1.shape)

pil_image = Image.open('/root/data/chazing_incubator/LiveTalking/data/avatars/avator_3/customvideo/images_nobg/00000000.png').convert('RGBA')
i2 = np.array(pil_image)
print(i2.shape)


def add_transparent_image(background, foreground, x_offset=None, y_offset=None):
    bg_h, bg_w, bg_channels = background.shape
    fg_h, fg_w, fg_channels = foreground.shape

    print(foreground.shape)

    assert bg_channels == 3, f'background image should have exactly 3 channels (RGB). found:{bg_channels}'
    assert fg_channels == 4, f'foreground image should have exactly 4 channels (RGBA). found:{fg_channels}'

    # center by default
    if x_offset is None: x_offset = (bg_w - fg_w) // 2
    if y_offset is None: y_offset = (bg_h - fg_h) // 2

    w = min(fg_w, bg_w, fg_w + x_offset, bg_w - x_offset)
    h = min(fg_h, bg_h, fg_h + y_offset, bg_h - y_offset)

    if w < 1 or h < 1: return

    # clip foreground and background images to the overlapping regions
    bg_x = max(0, x_offset)
    bg_y = max(0, y_offset)
    fg_x = max(0, x_offset * -1)
    fg_y = max(0, y_offset * -1)
    foreground = foreground[fg_y:fg_y + h, fg_x:fg_x + w]
    background_subsection = background[bg_y:bg_y + h, bg_x:bg_x + w]

    # separate alpha and color channels from the foreground image
    foreground_colors = foreground[:, :, :3]
    alpha_channel = foreground[:, :, 3] / 255  # 0-255 => 0.0-1.0

    # construct an alpha_mask that matches the image shape
    alpha_mask = alpha_channel[:, :, np.newaxis]

    # combine the background with the overlay image weighted by alpha
    composite = background_subsection * (1 - alpha_mask) + foreground_colors * alpha_mask

    # overwrite the section of the background image that has been updated
    background[bg_y:bg_y + h, bg_x:bg_x + w] = composite


def cv2PIL(img_cv):
    return Image.fromarray(cv2.cvtColor(img_cv,cv2.COLOR_BGR2RGBA))

def PIL2cv(img_pil):
    return cv2.cvtColor(np.asarray(img_pil),cv2.COLOR_RGBA2BGRA)


tmpn_me = datetime.now().strftime("%H%M%S")

print("wtf: ")

## method 1
# image = cv2.imread("/root/data/chazing_incubator/LiveTalking/data/avatars/avator_3/customvideo/images_nobg/00000000.png")
# h, w = image.shape[:2]
# test_bg = cv2.imread("/root/data/chazing_incubator/LiveTalking/data/background2.jpg")
# test_bg2 = cv2.resize(test_bg, (w, h))
# # dst = cv2.addWeighted(image, 1, test_bg2, 1, 1)
# dst = cv2.add(test_bg2, image)

x_offset = 0
y_offset = 0

path_background = "/root/data/chazing_incubator/LiveTalking/data/background.jpg"
path_overlay = "/root/data/chazing_incubator/LiveTalking/data/avatars/avator_3/customvideo/images_nobg/00000000.png"

rgb_data = cv2.imread(path_overlay, cv2.IMREAD_UNCHANGED)
print("fk1", rgb_data.shape)

rgb_data2 = cv2.imread("/root/data/chazing_incubator/LiveTalking/tmp2/153324.combine_frame.png")

alpha_data = rgb_data[:,:,3].copy()

input_image = cv2.cvtColor(rgb_data2, cv2.COLOR_RGB2RGBA)
input_image[:, :, 3] = alpha_data
cv2.imwrite(f"/root/data/chazing_incubator/LiveTalking/tmp/{tmpn_me}.cvtColor.png", input_image)

## method 2
# background = cv2.imread(path_background)
# overlay = cv2.imread(path_overlay)

# pil_image = Image.open(path_overlay).convert('RGBA')
# overlay = np.array(pil_image)

# cv2.imwrite(f"/root/data/chazing_incubator/LiveTalking/tmp/{tmpn_me}.overlay.png", overlay)

# img = background.copy()
# add_transparent_image(img, overlay, x_offset, y_offset)

# path_overlay = "/root/data/chazing_incubator/LiveTalking/tmp/141950.l2.png"

layer1 = Image.open(path_background).convert('RGBA')   # 底图背景
layer2 = cv2PIL(input_image)
#layer2 = Image.open(path_overlay).convert('RGBA')

# print(np.asarray(layer1).shape)
# print(np.asarray(layer2).shape)
_w, _h = layer2.size

layer1_new = layer1.resize((_w, _h), Image.LANCZOS)

final = Image.new("RGBA", layer2.size)             # 合成的image
final = Image.alpha_composite(final, layer1_new)
final = Image.alpha_composite(final, layer2)
# final = final.convert('RGB')

final.save(f"/root/data/chazing_incubator/LiveTalking/tmp/{tmpn_me}.png")

#cv2.imwrite(f"/root/data/chazing_incubator/LiveTalking/tmp/{tmpn_me}.png", img)

h, w = input_image.shape[:2]
test_bg2 = cv2.imread(path_background)
test_bg3 = cv2.resize(test_bg2, (w, h))
test_bg4 = cv2.cvtColor(test_bg3, cv2.COLOR_RGB2RGBA)
dst = cv2.add(test_bg4, input_image)
cv2.imwrite(f"/root/data/chazing_incubator/LiveTalking/tmp/{tmpn_me}.cv2.png", dst)


print(123)


