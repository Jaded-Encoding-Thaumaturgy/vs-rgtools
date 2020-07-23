__all__ = ['Blur', 'Sharpen', 'sbr', 'sbrH', 'sbrV', 'MinBlur', 'MinBlurH', 'MinBlurV']

import vapoursynth as vs
from .util import split, join, fallback, append_params, parse_planes, eval_planes, MinFilter, MedianClip
from functools import partial
from math import ceil, log2





def Blur(clip, radius=None, planes=None, mode='s', bmode='gauss', r=1):
    if not isinstance(clip, vs.VideoNode):
        raise TypeError('Blur: This is not a clip')
    
    numplanes = clip.format.num_planes
    
    radius = fallback(radius, r)
    
    radius = append_params(radius, numplanes)
    mode   = append_params(mode,   numplanes)
    bmode  = append_params(bmode,  numplanes)
    
    planes = parse_planes(planes, numplanes, 'Blur')
    planes = eval_planes(planes, radius, zero=0)
    
    if len(planes) == 0:
        return clip
    
    pr = [radius[x] for x in planes]
    pm = [mode[x]   for x in planes]
    pb = [bmode[x]  for x in planes]
    
    if any(len(set(p))>1 for p in (pr, pm, pb)):
        clips = split(clip)
        for x in planes:
            clips[x] = Blur_internal(clips[x], radius[x], [0], mode[x], bmode[x])
        return join(clips)
    
    return Blur_internal(clip, pr[0], planes, pm[0], pb[0])

def Blur_internal(clip, radius, planes, mode, bmode):
    fmt = clip.format
    isint = fmt.sample_type == vs.INTEGER
    numplanes = fmt.num_planes
    
    bmode = bmode.lower()
    mode = mode.lower()
    
    if 'g' not in bmode:
        if radius==1:
            if mode=='s':
                return RemoveGrain(clip, mode=[20,20,20], planes=planes)
            return clip.std.Convolution(matrix=[1,1,1], planes=planes, mode=mode)
        return clip.std.BoxBlur(planes=planes, hradius=0 if mode=='v' else radius, hpasses=1, vradius=0 if mode=='h' else radius, vpasses=1)
    
    if mode == 's':
        return RemoveGrainM(clip, [11,20], iter=radius, planes=planes)
    
    #This is my punishment for failing calculus
    gmatrix = [ [1,2,1],
                [1,3,4,3,1],
                [1,4,8,10,8,4,1],
                [1,5,13,22,26,22,13,5,1],
                [1,6,19,40,61,70,61,40,19,6,1],
                [1,7,26,65,120,171,192,171,120,65,26,7,1],
                [1,8,34,98,211,356,483,534,483,356,211,98,34,8,1],
                [x/10 for x in [1,9,43,140,343,665,1050,1373,1500,1373,1050,665,343,140,43,9,1]],
                [x/10 for x in [1,10,53,192,526,1148,2058,3088,3923,4246,3923,3088,2058,1148,526,192,53,10,1]],
                [x/100 for x in [1,11,64,255,771,1866,3732,6294,9069,11257,12092,11257,9069,6294,3732,1866,771,255,64,11,1]],
                [x/100 for x in [1,12,76,330,1090,2892,6369,11892,19095,26620,32418,34606,32418,26620,19095,11892,6369,2892,1090,330,76,12,1]],
                [x/100 for x in [1,13,89,418,1496,4312,10351,21153,37356,57607,78133,93644,99442,93644,78133,57607,37356,21153,10351,4312,1496,418,89,13,1]]
              ]
    bmatrix = [ [1,1,1],
                [1,2,3,2,1],
                [1,3,6,7,6,3,1],
                [1,4,10,16,19,16,10,4,1,],
                [1,5,15,30,45,51,45,30,15,5,1],
                [1,6,21,50,90,126,141,126,90,50,21,6,1],
                [1,7,28,77,161,266,357,393,357,266,161,77,28,7,1],
                [x/10 for x in [1,8,36,112,266,504,784,1016,1107,1016,784,504,266,112,36,8,1]],
                [x/10 for x in [1,9,45,156,414,882,1554,2304,2907,3139,2907,2304,1554,882,414,156,45,9,1]],
                [x/10 for x in [1,10,55,210,615,1452,2850,4740,6765,8350,8953,8350,6765,4740,2850,1452,615,210,55,10,1]],
                [x/100 for x in [1,11,66,275,880,2277,4917,9042,14355,19855,24068,25653,24068,19855,14355,9042,4917,2277,880,275,66,11,1]],
                [x/100 for x in [1,12,78,352,1221,3432,8074,16236,28314,43252,58278,69576,73789,69576,58278,43252,28314,16236,8074,3432,1221,352,78,12,1]]
              ]
    
    max_mtx = 7 if isint else 12
    
    clip = clip.std.Convolution(matrix=gmatrix[min(radius, max_mtx) - 1], planes=planes, mode=mode)
    radius -= max_mtx
    
    iter_conv = ceil(radius/max_mtx)
    
    for _ in range(iter_conv):
        clip = clip.std.Convolution(matrix=bmatrix[min(radius, max_mtx) - 1], planes=planes, mode=mode)
        radius -= max_mtx
    
    return clip



def Sharpen(clip, amountH=None, amountV=None, radius=1, planes=None, legacy=False, amount=1):
    if not isinstance(clip, vs.VideoNode):
        raise TypeError('Sharpen: This is not a clip')
    
    numplanes = clip.format.num_planes
    
    amountH = fallback(amountH, amount)
    amountH = append_params(amountH, numplanes)
    
    amountV = fallback(amountV, amountH)
    amountV = append_params(amountV, numplanes)
    
    radius = append_params(radius, numplanes)
    
    planes = parse_planes(planes, numplanes, 'Sharpen')
    planes = eval_planes(planes, [1 if (x, y) != (0, 0) else 0 for x, y in zip(amountH, amountV)], 0)
    
    if len(planes) == 0:
        return clip
    
    legacy = append_params(legacy, numplanes)
    
    ph = [amountH[x] for x in planes]
    pv = [amountV[x] for x in planes]
    pr = [radius[x]  for x in planes]
    pl = [legacy[x]  for x in planes]
    
    if any(len(set(p))>1 for p in (ph, pv, pr, pl)):
        clips = split(clip)
        for x in planes:
            clips[x] = Sharpen(clip, amountH[x], amountV[x], radius[x], [0], legacy[x])
        return join(clips)
    
    amountH, amountV, radius, legacy = -ph[0], -pv[0], pr[0], pl[0]
    
    if legacy:
        amountH = max(-1, min(amountH, log2(3)))
        amountV = max(-1, min(amountV, log2(3)))
    
    mode = 's'
    if amountV == 0:
        mode = 'h'
    if amountH == 0:
        mode = 'v'
        amountH = amountV
    
    if not 0 < radius <= (2 if mode is 's' else 12):
        raise ValueError('Sharpen: radius must be in the range of 1 - 2 for 2D processing or 1 - 12 for horizontal/vertical only processing') 
    
    relative_weight_h = 0.5 ** amountH / ((1 - 0.5 ** amountH) / 2)
    relative_weight_v = 0.5 ** amountV / ((1 - 0.5 ** amountV) / 2)
    
    if clip.format.sample_type == 0:
        matrices = [get_matrix_array(relative_weight_h, relative_weight_v, radius, x, mode) for x in range(1, 1024)]
        error = [sum([abs(x - min(max(round(x), -1023), 1023)) for x in set(matrix[1:])]) for matrix in matrices]
        matrix = [min(max(round(x), -1023), 1023) for x in matrices[error.index(min(error))]]
    else:
        base = 1000
        matrix = get_matrix_array(relative_weight_h, relative_weight_v, radius, base, mode)
        while max(matrix) > 1023:
            base /= 10
            matrix = get_matrix_array(relative_weight_h, relative_weight_v, radius, base, mode)
    
    if mode == 's':
        matrix = [matrix[x] for x in [(3,2,3,1,0,1,3,2,3), (8,7,6,7,8,5,4,3,4,5,2,1,0,1,2,5,4,3,4,5,8,7,6,7,8)][radius-1]]
    else:
        matrix = [matrix[x] for x in range(radius, 0, -1)] + matrix
    
    return clip.std.Convolution(matrix=matrix, planes=planes, mode=mode)

def get_matrix_array(rwh, rwv, radius, center, mode):
    matrix = [center]
    for x in range(radius):
        matrix.append(abs(matrix[-1]) / rwh)
    if mode == 's':
        for x in range(radius):
            for y in range(radius + 1):
                matrix.append(abs(matrix[(radius + 1) * x + y]) / rwv)
    return matrix



def MinBlur(clip, radius=None, planes=None, mode='s', bmode='gauss', range_in=None, memsize=1048576, opt=0, r=1):
    if not isinstance(clip, vs.VideoNode):
        raise TypeError('MinBlur: This is not a clip')
    
    numplanes = clip.format.num_planes
    
    radius = fallback(radius, r)
    radius = append_params(radius, numplanes)
    
    planes = parse_planes(planes, numplanes, 'MinBlur')
    planes = eval_planes(planes, radius, zero=-1)
    
    if len(planes) == 0:
        return clip
    
    sbr_radius = [1 if radius[x] == 0 else 0 for x in range(numplanes)]
    med_radius = [1 if radius[x] == 0 else radius[x] for x in range(numplanes)]
    
    RG11 = Blur(clip, radius, planes, mode, bmode)
    RG11 = sbr(RG11, sbr_radius, planes, mode, bmode)
    
    RG4 = Median(clip, med_radius, planes, mode, 1, range_in, memsize, opt)
    
    return MedianClip([clip, RG11, RG4], planes=planes)

MinBlurV = partial(MinBlur, mode='v')
MinBlurH = partial(MinBlur, mode='h')



def sbr(clip, radius=None, planes=None, mode='s', bmode='gauss', r=1):
    if not isinstance(clip, vs.VideoNode):
        raise TypeError('sbr: This is not a clip')
    
    numplanes = clip.format.num_planes
    
    radius = fallback(radius, r)
    
    radius = append_params(radius, numplanes)
    mode   = append_params(mode,   numplanes)
    bmode  = append_params(bmode,  numplanes)
    
    planes = parse_planes(planes, numplanes, 'sbr')
    planes = eval_planes(planes, radius, zero=0)
    
    pr = [radius[x] for x in planes]
    pm = [mode[x]   for x in planes]
    pb = [bmode[x]  for x in planes]
    
    if len(planes) == 0:
        return clip
    
    mixproc = numplanes > len(planes)
    mixproc = mixproc or any(len(set(p))>1 for p in (pr, pm, pb))
    
    if mixproc:
        clips = split(clip)
        for x in planes:
            clips[x] = MinFilter(clips[x], partial(Blur, radius=radius[x], mode=mode[x], bmode=bmode[x]))
        return join(clips)
    
    return MinFilter(clip, partial(Blur, radius=pr[0], planes=planes, mode=pm[0], bmode=pb[0]), planes=planes)

sbrV = partial(sbr, mode='v')
sbrH = partial(sbr, mode='h')
