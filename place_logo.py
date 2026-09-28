import cv2, numpy as np, sys
im = cv2.imread('images/1.jpg').astype(np.float32)
lg = cv2.imread('images/2.png', cv2.IMREAD_UNCHANGED)
H, W = im.shape[:2]; h, w = lg.shape[:2]
# top-face corners in photo (unnotched rectangle): left, back, right, front
dst = np.float32([[522,838],[1069,577],[1438,778],[881,1117]])
M = cv2.getPerspectiveTransform(np.float32([[0,0],[w,0],[w,h],[0,h]]), dst)

def polyfit(img, mask, deg=2):
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]].astype(np.float32)
    xx /= img.shape[1]; yy /= img.shape[0]
    terms = [xx**i * yy**j for i in range(deg+1) for j in range(deg+1-i)]
    A = np.stack([t[mask] for t in terms], 1)
    out = np.zeros_like(img)
    for c in range(3):
        coef, *_ = np.linalg.lstsq(A, img[..., c][mask], rcond=None)
        out[..., c] = sum(k*t for k, t in zip(coef, terms))
    return out

# ---- 1. erase old print: fit a smooth surface to clean chocolate pixels, fill the rest
face = cv2.warpPerspective(lg[:,:,3], M, (W, H)) > 128
face_in = cv2.erode(face.astype(np.uint8), np.ones((13,13))) > 0
x0,y0,bw,bh = cv2.boundingRect(face.astype(np.uint8))
sl = np.s_[y0:y0+bh, x0:x0+bw]
sub, fm = im[sl], face_in[sl]
B,G,R = sub[...,0], sub[...,1], sub[...,2]
# chocolate = yellowish tan: R-G ~25, G-B ~55; hearts are red/pink, label is pale
choc = fm & (np.abs((R-G)-26) < 10) & (np.abs((G-B)-58) < 14)
choc = cv2.erode(choc.astype(np.uint8), np.ones((7,7))) > 0
bg = polyfit(sub, choc, 3)
pm = fm & ~choc
pm = cv2.dilate(pm.astype(np.uint8), np.ones((5,5))) & fm
pm = cv2.GaussianBlur(pm.astype(np.float32), (0,0), 2)[..., None]
noise = cv2.GaussianBlur(np.random.randn(*sub.shape[:2]).astype(np.float32), (0,0), 0.7)[...,None] * 1.2
im[sl] = sub*(1-pm) + (bg+noise)*pm

# ---- 2. extract print from the flat design as a multiply layer
src = lg[:,:,:3].astype(np.float32)
b,g,r = src[...,0], src[...,1], src[...,2]
box = np.zeros((h,w), np.uint8); box[70:330, 55:465] = 1
pr = (((r-g) > 16) | (g < 170)).astype(np.uint8) & box
pr = cv2.morphologyEx(pr, cv2.MORPH_OPEN, np.ones((3,3)))
pr = cv2.morphologyEx(pr, cv2.MORPH_CLOSE, np.ones((9,9)))
n, lab, st, _ = cv2.connectedComponentsWithStats(pr)
pr = np.isin(lab, [i for i in range(1,n) if st[i,4] > 400]).astype(np.uint8)
ff = pr.copy(); cv2.floodFill(ff, None, (0,0), 1); pr |= (1-ff)   # fill holes
base = np.median(src[(box>0) & (pr==0)], 0)
ratio = np.clip(src / base, 0, 1)
a = cv2.GaussianBlur(pr.astype(np.float32), (0,0), 1)[...,None]
ratio = ratio*a + (1-a)
r_w = cv2.warpPerspective(ratio, M, (W, H), flags=cv2.INTER_CUBIC, borderValue=(1,1,1))
out = np.clip(im * r_w, 0, 255).astype(np.uint8)
cv2.imwrite(sys.argv[1] if len(sys.argv) > 1 else 'out.png', out)
cv2.imwrite('scratchpad/erased.png', im.astype(np.uint8)[500:1450,450:1500])
