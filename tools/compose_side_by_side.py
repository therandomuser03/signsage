#!/usr/bin/env python3
import sys
from pathlib import Path
from PIL import Image, ImageDraw

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
for name in sys.argv[2:]:
    gloss, label = name.split(':')
    avatar = Image.open(out / f'{gloss}_{label}.png')
    lm_path = out / f'{gloss}_{ {"start":"0.1","mid":"0.5","end":"0.9"}[label] }_lm.png'
    lm = Image.open(lm_path)
    h = avatar.height
    lm = lm.resize((int(lm.width * h / lm.height), h))
    canvas = Image.new('RGB', (avatar.width + lm.width, h), (17, 22, 29))
    canvas.paste(avatar, (0, 0))
    canvas.paste(lm, (avatar.width, 0))
    d = ImageDraw.Draw(canvas)
    d.text((8, 8), f'avatar · {gloss} {label}', fill=(255, 255, 255))
    canvas.save(out / f'{gloss}_{label}_side.png')
print('composed', len(sys.argv) - 2, 'images')
