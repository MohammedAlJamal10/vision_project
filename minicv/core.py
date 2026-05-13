"""minicv — compact image-processing library (NumPy + Matplotlib only)."""
import numpy as np
import os
from PIL import Image

# ── Validation ──────────────────────────────────────────────────────────────
def validate_image(img):
    if not isinstance(img, np.ndarray): raise TypeError(f"Expected ndarray, got {type(img).__name__}")
    if img.size == 0: raise ValueError("Empty image")
    if img.ndim not in (2,3): raise ValueError(f"Image must be 2-D or 3-D, got {img.ndim}-D")

def validate_grayscale(img):
    validate_image(img)
    if img.ndim != 2: raise ValueError(f"Expected 2-D grayscale, got shape {img.shape}")

def validate_kernel(k):
    if not isinstance(k, np.ndarray): raise TypeError("kernel must be ndarray")
    if k.ndim != 2: raise ValueError("kernel must be 2-D")
    if k.shape[0]%2==0 or k.shape[1]%2==0: raise ValueError("kernel dims must be odd")

# ── I/O ─────────────────────────────────────────────────────────────────────
def read_image(path, as_float=True):
    if not os.path.isfile(path): raise FileNotFoundError(f"Not found: {path}")
    img = np.asarray(Image.open(path))
    if as_float:
        img = img.astype(np.float64)/255.0 if img.dtype==np.uint8 else img.astype(np.float64)
    return img

def save_image(path, image, cmap=None):
    validate_image(image)
    img = np.clip(image.astype(np.float64), 0, 1)
    if img.ndim == 2:
        out = (img * 255).round().astype(np.uint8)
        Image.fromarray(out, mode="L").save(path)
    else:
        out = (img[..., :3] * 255).round().astype(np.uint8)
        Image.fromarray(out, mode="RGB").save(path)

def rgb_to_gray(img):
    validate_image(img)
    if img.ndim==2: return img.copy()
    return (0.2989*img[...,0]+0.5870*img[...,1]+0.1140*img[...,2]).astype(img.dtype)

def gray_to_rgb(img):
    validate_image(img)
    if img.ndim==3 and img.shape[2]==3: return img.copy()
    if img.ndim==3 and img.shape[2]==1: img=img[...,0]
    return np.stack([img,img,img], axis=-1)

# ── Normalisation & Clipping ─────────────────────────────────────────────────
def normalize(img, mode="minmax"):
    validate_image(img); x=img.astype(np.float64)
    if mode=="minmax":
        lo,hi=x.min(),x.max(); return np.zeros_like(x) if hi-lo<1e-12 else (x-lo)/(hi-lo)
    if mode=="zscore":
        mu,s=x.mean(),x.std(); return np.zeros_like(x) if s<1e-12 else (x-mu)/s
    if mode=="maxabs":
        m=np.abs(x).max(); return x if m<1e-12 else x/m
    raise ValueError(f"Unknown mode: {mode}")

def clip_image(img, lo=0.0, hi=1.0):
    validate_image(img)
    if lo>=hi: raise ValueError("lo must be < hi")
    return np.clip(img, lo, hi).astype(img.dtype)

# ── Padding ──────────────────────────────────────────────────────────────────
_PAD = {"zero":"constant","reflect":"reflect","replicate":"edge"}
def pad_image(img, ph, pw, mode="zero"):
    validate_image(img)
    if mode not in _PAD: raise ValueError(f"mode must be one of {set(_PAD)}")
    kw = {"constant_values":0} if mode=="zero" else {}
    pw2 = ((ph,ph),(pw,pw)) if img.ndim==2 else ((ph,ph),(pw,pw),(0,0))
    return np.pad(img, pw2, mode=_PAD[mode], **kw)

# ── Convolution ──────────────────────────────────────────────────────────────
def _conv_single(img, kernel, mode):
    H,W = img.shape; kH,kW = kernel.shape; pH,pW = kH//2,kW//2
    padded = pad_image(img, pH, pW, mode)
    s0,s1 = padded.strides
    wins = np.lib.stride_tricks.as_strided(padded,(H,W,kH,kW),(s0,s1,s0,s1))
    return np.einsum("hwij,ij->hw", wins, kernel, optimize=True)

def convolve2d(img, kernel, padding_mode="zero"):
    validate_image(img); validate_kernel(kernel)
    k = np.flip(kernel).astype(np.float64); img=img.astype(np.float64)
    if img.ndim==2: return _conv_single(img,k,padding_mode)
    return np.stack([_conv_single(img[...,c],k,padding_mode) for c in range(img.shape[2])],axis=-1)

# ── Kernels ───────────────────────────────────────────────────────────────────
def make_box_kernel(size):
    if size%2==0: raise ValueError("size must be odd")
    return np.ones((size,size),dtype=np.float64)/(size*size)

def make_gaussian_kernel(size, sigma):
    if size%2==0: raise ValueError("size must be odd")
    if sigma<=0: raise ValueError("sigma must be > 0")
    h=size//2; ax=np.arange(-h,h+1,dtype=np.float64)
    xx,yy=np.meshgrid(ax,ax)
    k=np.exp(-(xx**2+yy**2)/(2*sigma**2)); return k/k.sum()

# ── Filters ───────────────────────────────────────────────────────────────────
def mean_filter(img, size=3, padding_mode="reflect"):
    return convolve2d(img, make_box_kernel(size), padding_mode)

def gaussian_filter(img, size=5, sigma=1.0, padding_mode="reflect"):
    return convolve2d(img, make_gaussian_kernel(size,sigma), padding_mode)

def _median_ch(img, size, mode):
    H,W=img.shape; h=size//2
    p=pad_image(img,h,h,mode); s0,s1=p.strides
    wins=np.lib.stride_tricks.as_strided(p,(H,W,size,size),(s0,s1,s0,s1))
    return np.median(wins.reshape(H,W,size*size),axis=-1)

def median_filter(img, size=3, padding_mode="reflect"):
    validate_image(img)
    if size%2==0: raise ValueError("size must be odd")
    img=img.astype(np.float64)
    if img.ndim==2: return _median_ch(img,size,padding_mode)
    return np.stack([_median_ch(img[...,c],size,padding_mode) for c in range(img.shape[2])],axis=-1)

_SX=np.array([[-1,0,1],[-2,0,2],[-1,0,1]],dtype=np.float64)
_SY=np.array([[-1,-2,-1],[0,0,0],[1,2,1]],dtype=np.float64)

def sobel_gradients(img, padding_mode="reflect"):
    validate_grayscale(img)
    return _conv_single(img.astype(np.float64),np.flip(_SX),padding_mode), \
           _conv_single(img.astype(np.float64),np.flip(_SY),padding_mode)

def gradient_magnitude(img, padding_mode="reflect"):
    gx,gy=sobel_gradients(img,padding_mode); return np.sqrt(gx**2+gy**2)

# ── Thresholding ─────────────────────────────────────────────────────────────
def global_threshold(img, t):
    validate_grayscale(img); return (img>=t).astype(np.float64)

def otsu_threshold(img):
    validate_grayscale(img)
    x=img.astype(np.float64)
    xi=(np.clip(np.round(x*255 if x.max()<=1.0 else x),0,255)).astype(np.int32)
    L=256; N=xi.size; p=np.bincount(xi.ravel(),minlength=L).astype(np.float64)/N
    w=np.cumsum(p); mu=np.cumsum(p*np.arange(L)); mt=mu[-1]
    w1=1-w
    mu0=np.divide(mu,w,out=np.zeros_like(mu),where=w>0)
    mu1=np.divide(mt-mu,w1,out=np.zeros_like(mu),where=w1>0)
    sb=w*w1*(mu0-mu1)**2; t=int(np.argmax(sb))/255.0
    return (x>=t).astype(np.float64),t

def adaptive_mean_threshold(img, block_size=11, C=0.02):
    validate_grayscale(img)
    if block_size%2==0: raise ValueError("block_size must be odd")
    return (img>=mean_filter(img,block_size,"reflect")-C).astype(np.float64)

def adaptive_gaussian_threshold(img, block_size=11, sigma=2.0, C=0.02):
    validate_grayscale(img)
    if block_size%2==0: raise ValueError("block_size must be odd")
    return (img>=gaussian_filter(img,block_size,sigma,"reflect")-C).astype(np.float64)

# ── Histogram ─────────────────────────────────────────────────────────────────
def compute_histogram(img, bins=256, density=False):
    validate_grayscale(img)
    r=(float(img.min()),float(img.max())); r=(r[0],r[0]+1e-9) if r[0]==r[1] else r
    h,e=np.histogram(img.ravel(),bins=bins,range=r,density=density)
    return h.astype(np.float64),e

def histogram_equalization(img):
    validate_grayscale(img)
    x=img.astype(np.float64)
    xi=np.clip(np.round(x*255 if x.max()<=1.0 else x),0,255).astype(np.int32)
    h=np.bincount(xi.ravel(),minlength=256).astype(np.float64)
    cdf=np.cumsum(h); cm=cdf[cdf>0][0]; d=xi.size-cm
    eq=np.clip((cdf-cm)/d if d>0 else cdf*0,0,1)
    return eq[xi]

# ── Bit-plane ─────────────────────────────────────────────────────────────────
def _to_u8(img):
    if img.dtype==np.uint8: return img
    x=img.astype(np.float64)
    return np.clip(np.round(x*255 if x.max()<=1.0 else x),0,255).astype(np.uint8)

def extract_bit_plane(img, bit):
    validate_grayscale(img)
    if not 0<=bit<=7: raise ValueError("bit must be in [0,7]")
    return ((_to_u8(img)>>bit)&1).astype(np.uint8)

def all_bit_planes(img): return [extract_bit_plane(img,b) for b in range(8)]

# ── Resize ────────────────────────────────────────────────────────────────────
def _resize_ch(img, nh, nw, method):
    H,W=img.shape
    rr=np.linspace(0,H-1,nh); cc=np.linspace(0,W-1,nw)
    rg,cg=np.meshgrid(rr,cc,indexing="ij")
    if method=="nearest":
        return img[np.clip(np.round(rg).astype(int),0,H-1),
                   np.clip(np.round(cg).astype(int),0,W-1)]
    r0=np.floor(rg).astype(int); c0=np.floor(cg).astype(int)
    r1=np.clip(r0+1,0,H-1); c1=np.clip(c0+1,0,W-1); r0=np.clip(r0,0,H-1); c0=np.clip(c0,0,W-1)
    dr=rg-np.floor(rg); dc=cg-np.floor(cg)
    return ((1-dr)*(1-dc)*img[r0,c0]+(1-dr)*dc*img[r0,c1]+
            dr*(1-dc)*img[r1,c0]+dr*dc*img[r1,c1])

def resize(img, new_h, new_w, method="bilinear"):
    validate_image(img)
    if method not in ("nearest","bilinear"): raise ValueError(f"Unknown method: {method}")
    img=img.astype(np.float64)
    if img.ndim==2: return _resize_ch(img,new_h,new_w,method)
    return np.stack([_resize_ch(img[...,c],new_h,new_w,method) for c in range(img.shape[2])],axis=-1)

# ── Rotate ────────────────────────────────────────────────────────────────────
def _rotate_ch(img, angle, method, fv):
    H,W=img.shape; cx,cy=(W-1)/2,(H-1)/2
    t=np.deg2rad(angle); ct,st=np.cos(t),np.sin(t)
    or_,oc_=np.meshgrid(np.arange(H),np.arange(W),indexing="ij")
    dc=oc_-cx; dr=or_-cy
    sr=(-st*dc+ct*dr)+cy; sc=(ct*dc+st*dr)+cx
    out=np.full((H,W),fv,dtype=np.float64)
    if method=="nearest":
        r=np.round(sr).astype(int); c=np.round(sc).astype(int)
        v=(r>=0)&(r<H)&(c>=0)&(c<W)
        out[or_[v],oc_[v]]=img[r[v],c[v]]
    else:
        r0=np.floor(sr).astype(int); c0=np.floor(sc).astype(int)
        v=(r0>=0)&(r0<H)&(c0>=0)&(c0<W)
        vr,vc=or_[v],oc_[v]; r0v,c0v=r0[v],c0[v]
        r1v=np.clip(r0v+1,0,H-1); c1v=np.clip(c0v+1,0,W-1)
        drf=sr[v]-r0v; dcf=sc[v]-c0v
        out[vr,vc]=((1-drf)*(1-dcf)*img[r0v,c0v]+(1-drf)*dcf*img[r0v,c1v]+
                    drf*(1-dcf)*img[r1v,c0v]+drf*dcf*img[r1v,c1v])
    return out

def rotate(img, angle, method="bilinear", fill_value=0.0):
    validate_image(img); img=img.astype(np.float64)
    if img.ndim==2: return _rotate_ch(img,angle,method,fill_value)
    return np.stack([_rotate_ch(img[...,c],angle,method,fill_value) for c in range(img.shape[2])],axis=-1)

# ── Translate ─────────────────────────────────────────────────────────────────
def translate(img, tx, ty, fill_value=0.0):
    validate_image(img); img=img.astype(np.float64)
    H,W=img.shape[:2]; out=np.full_like(img,fill_value)
    sr0=max(0,-ty); sr1=min(H,H-ty); sc0=max(0,-tx); sc1=min(W,W-tx)
    dr0=max(0,ty); dc0=max(0,tx)
    if sr0<sr1 and sc0<sc1:
        out[dr0:dr0+(sr1-sr0),dc0:dc0+(sc1-sc0)]=img[sr0:sr1,sc0:sc1]
    return out

# ── Flip (extra augmentation primitive) ──────────────────────────────────────
def flip(img, axis="horizontal"):
    validate_image(img)
    if axis=="horizontal": return img[:,::-1].copy()
    if axis=="vertical":   return img[::-1,:].copy()
    raise ValueError("axis must be 'horizontal' or 'vertical'")

# ── Feature Descriptors ───────────────────────────────────────────────────────
def mean_intensity(img):   validate_image(img); return float(img.mean())
def std_intensity(img):    validate_image(img); return float(img.std())

def histogram_descriptor(img, bins=64):
    validate_image(img); img=img.astype(np.float64)
    if img.ndim==2:
        h,_=np.histogram(img.ravel(),bins=bins,range=(img.min(),max(img.max(),img.min()+1e-9)),density=True)
        return h.astype(np.float64)
    parts=[]
    for c in range(img.shape[2]):
        ch=img[...,c]
        h,_=np.histogram(ch.ravel(),bins=bins,range=(ch.min(),max(ch.max(),ch.min()+1e-9)),density=True)
        parts.append(h.astype(np.float64))
    return np.concatenate(parts)

def gradient_magnitude_histogram(img, bins=32, padding_mode="reflect"):
    validate_grayscale(img)
    mag=gradient_magnitude(img,padding_mode)
    h,_=np.histogram(mag.ravel(),bins=bins,range=(0,2)); s=h.sum()
    return (h/s if s>0 else h).astype(np.float64)

def edge_density(img, threshold=0.1, padding_mode="reflect"):
    validate_grayscale(img)
    return float((gradient_magnitude(img,padding_mode)>threshold).sum())/img.size

def describe_image(img, hist_bins=32, grad_bins=16):
    validate_image(img)
    gray=rgb_to_gray(img) if img.ndim==3 else img.astype(np.float64)
    if gray.max()>1.0: gray=gray/255.0
    return np.concatenate([
        [mean_intensity(img), std_intensity(img)],
        histogram_descriptor(gray,hist_bins),
        gradient_magnitude_histogram(gray,grad_bins),
        [edge_density(gray)],
    ])

# ── Canvas ops ────────────────────────────────────────────────────────────────
def stack_horizontal(imgs, fill=0.0):
    if not imgs: raise ValueError("empty list")
    mh=max(i.shape[0] for i in imgs); out=[]
    for i in imgs:
        i=i.astype(np.float64); d=mh-i.shape[0]
        if d>0:
            p=np.full((d,i.shape[1])if i.ndim==2 else (d,i.shape[1],i.shape[2]),fill)
            i=np.concatenate([i,p],0)
        out.append(i)
    return np.concatenate(out,1)

def stack_vertical(imgs, fill=0.0):
    if not imgs: raise ValueError("empty list")
    mw=max(i.shape[1] for i in imgs); out=[]
    for i in imgs:
        i=i.astype(np.float64); d=mw-i.shape[1]
        if d>0:
            p=np.full((i.shape[0],d)if i.ndim==2 else (i.shape[0],d,i.shape[2]),fill)
            i=np.concatenate([i,p],1)
        out.append(i)
    return np.concatenate(out,0)

def blend(a, b, alpha=0.5):
    if a.shape!=b.shape: raise ValueError("shapes must match")
    return alpha*a.astype(np.float64)+(1-alpha)*b.astype(np.float64)

def crop(img, x0, y0, x1, y1):
    validate_image(img); H,W=img.shape[:2]
    if not(0<=x0<x1<=W and 0<=y0<y1<=H): raise ValueError("invalid crop")
    return img[y0:y1,x0:x1]

# ── Drawing ───────────────────────────────────────────────────────────────────
def draw_point(c, x, y, color, thickness=1):
    H,W=c.shape[:2]; h=thickness//2
    c[max(0,y-h):min(H,y+h+1), max(0,x-h):min(W,x+h+1)]=color; return c

def draw_line(c, x0, y0, x1, y1, color, thickness=1):
    dx,dy=abs(x1-x0),abs(y1-y0); sx=1 if x0<x1 else -1; sy=1 if y0<y1 else -1; err=dx-dy
    while True:
        draw_point(c,x0,y0,color,thickness)
        if x0==x1 and y0==y1: break
        e2=2*err
        if e2>-dy: err-=dy; x0+=sx
        if e2<dx:  err+=dx; y0+=sy
    return c

def draw_rectangle(c, x0, y0, x1, y1, color, thickness=1, filled=False):
    if filled:
        H,W=c.shape[:2]
        c[max(0,min(y0,y1)):min(H,max(y0,y1)+1),max(0,min(x0,x1)):min(W,max(x0,x1)+1)]=color
    else:
        for (a,b,d,e) in [(x0,y0,x1,y0),(x0,y1,x1,y1),(x0,y0,x0,y1),(x1,y0,x1,y1)]:
            draw_line(c,a,b,d,e,color,thickness)
    return c

def draw_polygon(c, pts, color, thickness=1, filled=False):
    if len(pts)<2: raise ValueError("need >= 2 points")
    n=len(pts)
    for i in range(n): draw_line(c,pts[i][0],pts[i][1],pts[(i+1)%n][0],pts[(i+1)%n][1],color,thickness)
    return c

__version__ = "1.0.0"
