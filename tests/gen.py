"""Synthetic test drawings (schematic + ladder) so tests need no external files."""
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

def _text(im, items, size):
    pil = Image.fromarray(im); d = ImageDraw.Draw(pil); F = ImageFont.truetype(FONT, size)
    for xy, t in items: d.text(xy, t, font=F, fill=(0, 0, 0))
    return np.array(pil)

def circuit_image():
    G = (40, 140, 60); im = np.full((430, 800, 3), 255, np.uint8)
    def ln(*p): cv2.polylines(im, [np.array(p, np.int32)], False, G, 3)
    def zig(a, b, vert):
        pts = [a]
        for i in range(1, 6): pts.append((a[0] + (b[0]-a[0])*i//6 + (14*(1 if i % 2 else -1) if vert else 0), a[1] + (b[1]-a[1])*i//6 + (0 if vert else 14*(1 if i % 2 else -1))))
        ln(*pts, b)
    ln((230, 100), (355, 100)); zig((355, 100), (400, 100), False); ln((400, 100), (573, 100), (573, 225)); zig((573, 225), (573, 275), True); ln((573, 275), (573, 330))
    ln((230, 100), (230, 225)); zig((230, 225), (230, 275), True); ln((230, 275), (230, 330))
    ln((118, 330), (687, 330)); ln((118, 330), (118, 285)); ln((118, 238), (118, 180), (358, 180)); ln((402, 180), (687, 180), (687, 238)); ln((687, 285), (687, 330))
    for c in ((380, 180), (118, 262), (687, 262)): cv2.circle(im, c, 22, G, 3)
    ln((118, 275), (118, 250)); ln((110, 258), (118, 248), (126, 258)); ln((687, 250), (687, 275)); ln((679, 267), (687, 277), (695, 267))
    ln((368, 180), (374, 180)); ln((386, 180), (392, 180)); ln((389, 174), (389, 186))
    ln((395, 330), (395, 350))
    for y, w in ((350, 22), (360, 15), (370, 8)): ln((395 - w, y), (395 + w, y))
    return _text(im, [((338, 62), "10Ω"), ((250, 240), "2Ω"), ((525, 240), "4Ω"), ((30, 240), "2A"), ((715, 240), "7A"), ((415, 150), "2V")], 22)

def ladder_image():
    K = (0, 0, 0); im = np.full((430, 800, 3), 255, np.uint8)
    def ln(*p, w=3): cv2.polylines(im, [np.array(p, np.int32)], False, K, w)
    def no(x, y): ln((x - 10, y - 15), (x - 10, y + 15)); ln((x + 10, y - 15), (x + 10, y + 15))
    def nc(x, y): no(x, y); ln((x - 10, y + 12), (x + 10, y - 12), w=2)
    def coil(x, y): cv2.ellipse(im, (x - 2, y), (8, 15), 0, 90, 270, K, 3); cv2.ellipse(im, (x + 2, y), (8, 15), 0, -90, 90, K, 3)
    ln((100, 50), (100, 380), w=4); ln((700, 50), (700, 380), w=4)
    ln((100, 120), (200, 120)); ln((220, 120), (270, 120)); ln((150, 120), (150, 200)); ln((270, 120), (270, 200)); ln((150, 200), (200, 200)); ln((220, 200), (270, 200))
    ln((270, 120), (390, 120)); ln((410, 120), (560, 120)); ln((600, 120), (700, 120)); no(210, 120); no(210, 200); nc(400, 120); coil(580, 120)
    ln((100, 300), (240, 300)); ln((260, 300), (560, 300)); ln((600, 300), (700, 300)); no(250, 300); coil(580, 300)
    return _text(im, [((195, 72), "X0"), ((195, 152), "Y0"), ((385, 72), "X1"), ((565, 72), "Y0"), ((235, 252), "Y0"), ((565, 252), "Y1"), ((40, 108), "1"), ((40, 288), "2")], 18)
