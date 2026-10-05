from PIL import Image
from pathlib import Path
ps=sorted(Path('tmp/pdfs/qa').glob('page_*.png'))
for start in range(0,len(ps),4):
    canvas=Image.new('RGB',(1000,1460),'#dce3e8')
    for j,path in enumerate(ps[start:start+4]):
        im=Image.open(path); im.thumbnail((485,700))
        canvas.paste(im,((j%2)*500+7,(j//2)*730+20))
    canvas.save(f'tmp/pdfs/qa/contact_{start//4+1}.png')
