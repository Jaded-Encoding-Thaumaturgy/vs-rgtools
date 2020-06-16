import vapoursynth as vs
from vsutil import split, join, fallback
from functools import partial
from math import ceil





def Repair(clip, repairclip, mode=2, planes=None):
    if not isinstance(clip, vs.VideoNode):
        raise TypeError('Repair: "clip" is not a clip')
    if not isinstance(repairclip, vs.VideoNode):
        raise TypeError('Repair: "repairclip" is not a clip')
    
    fmt = clip.format
    numplanes = fmt.num_planes
    
    rfmt = repairclip.format
    rnumplanes = rfmt.num_planes
    
    mode = append_params(mode, numplanes)[:numplanes]
    
    planes = parse_planes(planes, numplanes, 'Repair')
    mode = [max(mode[x], 0) if x in planes else 0 for x in range(numplanes)]
    
    if numplanes > rnumplanes:
        mode = [mode[i] if i > rnumplanes else 0 for i in range(len(numplanes))]
    
    if max(mode)==0:
        return clip
    repairclip = repairclip.resize.Bicubic(format=fmt.id)
    
################################################
### Mode 1 - Use Minimum/Maximum for float clips
    planes = []
    if fmt.sample_type==vs.FLOAT:
        for i in range(numplanes):
            if mode[i] in (1, 11):
                mode[i] = 0
                planes = planes + [i]
    
    clip = rep1(clip, repairclip, planes)
################################################
    
    return get_namespace(clip).Repair(clip, repairclip, mode)



def RemoveGrain(clip, mode=2, planes=None):
    if not isinstance(clip, vs.VideoNode):
        raise TypeError('RemoveGrain: This is not a clip')
    
    fmt = clip.format
    numplanes = fmt.num_planes
    isint = fmt.sample_type==vs.INTEGER
    
    mode = append_params(mode, numplanes)[:numplanes]
    
    planes = parse_planes(planes, numplanes, 'RemoveGrain')
    mode = [max(mode[x], 0) if x in planes else 0 for x in range(numplanes)]
    
    planes_rg4  = []
    planes_rg11 = []
    planes_rg19 = []
    planes_rg20 = []
    
######################################################
### Mode 4 - Use std.Median() ########################
    for i in range(numplanes):
        if mode[i]==4:
            planes_rg4 += [i]
            mode[i] = 0
    clip = rg4 (clip, planes_rg4)
######################################################


#########################################################
### Mode 11/12 - Use std.Convolution([1,2,1,2,4,2,1,2,1])
    for i in range(numplanes):
        if mode[i] in (11, 12):
            planes_rg11 += [i]
            mode[i] = 0
    clip = rg11(clip, planes_rg11)
#########################################################


######################################################
### Mode 19 - Use std.Convolution([1,1,1,1,0,1,1,1,1])
    for i in range(numplanes):
        if mode[i]==19:
            planes_rg19 += [i]
            mode[i] = 0
    clip = rg19(clip, planes_rg19)
######################################################


######################################################
### Mode 20 - Use std.Convolution([1,1,1,1,1,1,1,1,1])
    for i in range(numplanes):
        if mode[i]==20:
            planes_rg20 += [i]
            mode[i] = 0
    clip = rg20(clip, planes_rg20)
######################################################
    
    if max(mode)==0:
        return clip
    return get_namespace(clip).RemoveGrain(clip, mode)



def Clense(clip, previous=None, next=None, planes=None, grey=False):
    if not isinstance(clip, vs.VideoNode):
        raise TypeError('Clense: This is not a clip')
    if next is not None:
        if not isinstance(next, vs.VideoNode):
            raise TypeError('Clense: This is not a clip')
    if previous is not None:
        if not isinstance(previous, vs.VideoNode):
            raise TypeError('Clense: This is not a clip')
    
    if grey:
        planes = 0
    
###################################################################
### Use std.Expr for integer clips ################################
    if clip.format.bits_per_sample < 32:
        previous = fallback(previous, clip)
        previous = previous[0] + previous[:-1]
        next  = fallback(next, clip)
        next  = next[1:] + next[-1]
        return MedianClip([clip, previous, next], planes=planes)
###################################################################
    
    return get_namespace(clip).Clense(clip, next, previous, planes)



def ForwardClense(clip, planes=None, grey=False):
    if not isinstance(clip, vs.VideoNode):
        raise TypeError('ForwardClense: This is not a clip')
    if grey:
        planes = 0
    return get_namespace(clip).ForwardClense(clip, planes)



def BackwardClense(clip, planes=None, grey=False):
    if not isinstance(clip, vs.VideoNode):
        raise TypeError('BackwardClense: This is not a clip')
    if grey:
        planes = 0
    return get_namespace(clip).BackwardClense(clip, planes)



def VerticalCleaner(clip, mode=1, planes=None):
    if not isinstance(clip, vs.VideoNode):
        raise TypeError('VerticalCleaner: This is not a clip')
    
    numplanes = clip.format.num_planes
    
    mode = append_params(mode, numplanes)[:numplanes]
    
    planes = parse_planes(planes, numplanes, 'VerticalCleaner')
    mode = [max(mode[x], 0) if x in planes else 0 for x in range(numplanes)]
    
    return get_namespace(clip).VerticalCleaner(clip, mode)





def RemoveGrainM(clip, mode=2, modeu=None, modev=None, iter=None, planes=None):
    if not isinstance(clip, vs.VideoNode):
        raise TypeError('RemoveGrainM: This is not a clip')
    
    numplanes = clip.format.num_planes
    planes = parse_planes(planes, numplanes, 'RemoveGrainM')
   
    mode = [mode] if isinstance(mode, int) else mode
    modeu = [0] if numplanes < 2 else fallback([modeu] if isinstance(modeu, int) else modeu, mode )
    modev = [0] if numplanes < 3 else fallback([modev] if isinstance(modev, int) else modev, modeu)
    
    iter = fallback(iter, [len(x) for x in (mode,modeu,modev)])
    iter = append_params(iter, 3)
    iter = [iter[x] if x in planes else 0 for x in range(3)]
    
    mode  = [0] if iter[0] is 0 else append_params(mode , iter[0])[:iter[0]]
    modeu = [0] if iter[1] is 0 else append_params(modeu, iter[1])[:iter[1]]
    modev = [0] if iter[2] is 0 else append_params(modev, iter[2])[:iter[2]]
    
    iterall = max(iter[:numplanes])
    
    while len(mode ) < iterall:
        mode .append(0)
    while len(modeu) < iterall:
        modeu.append(0)
    while len(modev) < iterall:
        modev.append(0)
    
    for x in range(iterall):
        clip = RemoveGrain(clip, [mode[x], modeu[x], modev[x]])
    
    return clip





def Median(clip, radius=None, planes=None, mode='s', vcmode=1, range_in=None, memsize=1048576, opt=0, r=1):
    if not isinstance(clip, vs.VideoNode):
        raise TypeError('Median: This is not a clip')
    
    f = clip.format
    bits = f.bits_per_sample
    numplanes = f.num_planes
    range_in = fallback(range_in, f.color_family is vs.RGB)
    
    planes = parse_planes(planes, numplanes, 'Median')
    
    radius = fallback(radius, r)
    
    radius = append_params(radius, numplanes)[:numplanes]
    mode   = append_params(mode,   numplanes)[:numplanes]
    vcmode = append_params(vcmode, numplanes)[:numplanes]
    
    radius = [radius[x] if x in planes else 0 for x in range(numplanes)]
    vcmode = [vcmode[x] if x in planes else 0 for x in range(numplanes)]
    aplanes = [3 if radius[x] > 0 else 1 for x in range(numplanes)]
    
    pr = [radius[x] for x in planes]
    pm = [mode[x]   for x in planes]
    
    if max(radius) < 1:
        return clip
    
    mixproc = ('h' in pm or 'v' in pm) and max(pr) > 1 and numplanes - len(planes) != 0 # no planes parameter in average.Median
    mixproc = mixproc or any(len(set(p))>1 for p in (pr, pm))
    
    if mixproc:
        clips = split(clip)
        return join([Median_internal(clips[x], radius[x], [3], mode[x], vcmode[x], range_in, [0,1,1][x], memsize, opt) for x in range(numplanes)])
    
    return Median_internal(clip, pr[0], aplanes, pm[0], vcmode, range_in, 0, memsize, opt)

def Median_internal(clip, radius, aplanes, mode, vcmode, range_in, cal, memsize, opt):
    core = vs.core
    
    fmt = clip.format
    numplanes = fmt.num_planes
    
    mode = mode.lower()
    
    vplanes = avs_to_vs(aplanes)
    
    if radius == 0:
        return clip
    
    if radius == 1:
        if mode == 's':
            return core.std.Median(clip, planes=vplanes)
        if mode == 'v':
            return VerticalCleaner(clip, mode=vcmode, planes=vplanes)
        return VerticalCleaner(core.std.Transpose(clip), mode=vcmode, planes=vplanes).std.Transpose()
    
    if mode == 's':
        if fmt.sample_type==vs.FLOAT:
            down = Depth(clip, vplanes, range_in, cal)
            medd = core.ctmf.CTMF(down, radius=radius, memsize=memsize, opt=opt, planes=vplanes)
            medu = Depth(medd, vplanes, range_in, cal, fmt)
            return core.std.Expr([clip, medu, down, medd], ['z a = x y ?' if x in vplanes else '' for x in range(numplanes)])
        
        return core.ctmf.CTMF(clip, radius=radius, memsize=memsize, opt=opt, planes=vplanes)
    
    if mode == 'v':
        clips = [core.resize.Point(clip, src_top =x) for x in range(-radius, radius+1)]
    else:
        clips = [core.resize.Point(clip, src_left=x) for x in range(-radius, radius+1)]
    
    return core.average.Median(clips)



def Blur(clip, radius=None, planes=None, mode='s', blur='gauss', r=1):
    if not isinstance(clip, vs.VideoNode):
        raise TypeError('Blur: This is not a clip')
    
    numplanes = clip.format.num_planes
    
    planes = parse_planes(planes, numplanes, 'Blur')
    
    radius = fallback(radius, r)
    
    radius = append_params(radius, numplanes)[:numplanes]
    mode   = append_params(mode,   numplanes)[:numplanes]
    blur   = append_params(blur,   numplanes)[:numplanes]
    
    radius = [radius[x] if x in planes else 0 for x in range(numplanes)]
    aplanes = [3 if radius[x] > 0 else 1 for x in range(numplanes)]
    
    pr = [radius[x] for x in planes]
    pm = [mode[x]   for x in planes]
    pb = [blur[x]   for x in planes]
    
    if max(radius) < 1:
        return clip
    
    mixproc = any(len(set(p))>1 for p in (pr, pm, pb))
    
    if mixproc:
        clips = split(clip)
        return join([Blur_internal(clips[x], radius[x], [3], mode[x], blur[x]) for x in range(numplanes)])
    
    return Blur_internal(clip, pr[0], aplanes, pm[0], pb[0])

def Blur_internal(clip, radius, aplanes, mode, blur):
    core = vs.core
    
    fmt = clip.format
    isint = fmt.sample_type==vs.INTEGER
    numplanes = fmt.num_planes
    
    blur = blur.lower()
    mode = mode.lower()
    
    vplanes = avs_to_vs(aplanes)
    
    if radius==0:
        return clip
    
    if 'g' not in blur:
        if radius==1:
            if mode=='s':
                return RemoveGrain(clip, mode=[20,20,20], planes=vplanes)
            return core.std.Convolution(clip, matrix=[1,1,1], planes=vplanes, mode=mode)
        return core.std.BoxBlur(clip, planes=vplanes, hradius=0 if mode=='v' else radius, hpasses=1, vradius=0 if mode=='h' else radius, vpasses=1)
    
    if mode=='s':
        return RemoveGrainM(clip, [11,20], iter=radius, planes=vplanes)
    
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
    
    clip = core.std.Convolution(clip, matrix=gmatrix[min(radius, max_mtx) - 1], planes=vplanes, mode=mode)
    radius -= max_mtx
    
    iter_conv = ceil(radius/max_mtx)
    
    for _ in range(iter_conv):
        clip = core.std.Convolution(clip, matrix=bmatrix[min(radius, max_mtx) - 1], planes=vplanes, mode=mode)
        radius -= max_mtx
    
    return clip



def MinBlur(clip, radius=None, planes=None, mode='s', blur='gauss', range_in=None, memsize=1048576, opt=0, r=1):
    if not isinstance(clip, vs.VideoNode):
        raise TypeError('MinBlur: This is not a clip')
    
    numplanes = clip.format.num_planes
    
    planes = parse_planes(planes, numplanes, 'MinBlur')
    
    radius = fallback(radius, r)
    radius = append_params(radius, numplanes)[:numplanes]
    radius = [radius[x] if x in planes else 0 for x in range(numplanes)]
    
    aplanes = [3 if radius[x] >= 0 else 1 for x in range(numplanes)]
    
    if max(radius) < 0:
        return clip
    
    sbr_radius = [1 if radius[x] is 0 else 0 for x in range(numplanes)]
    med_radius = [1 if radius[x] is 0 else radius[x] for x in range(numplanes)]
    
    RG11 = Blur(clip, radius, avs_to_vs(aplanes), mode, blur)
    RG11 = sbr(RG11, sbr_radius, avs_to_vs(aplanes), mode, blur)
    
    RG4 = Median(clip, med_radius, avs_to_vs(aplanes), mode, 1, range_in, memsize, opt)
    
    return MedianClip([clip, RG11, RG4], planes=planes)

MinBlurV = partial(MinBlur, mode='v')
MinBlurH = partial(MinBlur, mode='h')



def sbr(clip, radius=None, planes=None, mode='s', blur='gauss', r=1):
    if not isinstance(clip, vs.VideoNode):
        raise TypeError('sbr: This is not a clip')
    
    numplanes = clip.format.num_planes
    
    planes = parse_planes(planes, numplanes, 'sbr')
    
    radius = fallback(radius, r)
    
    radius = append_params(radius, numplanes)[:numplanes]
    mode   = append_params(mode,   numplanes)[:numplanes]
    blur   = append_params(blur,   numplanes)[:numplanes]
    
    radius = [radius[x] if x in planes else -1 for x in range(numplanes)]
    aplanes = [3 if radius[x] >= 0 else 1 for x in range(numplanes)]
    
    pr = [radius[x] for x in planes]
    pm = [mode[x] for x in planes]
    pb = [blur[x] for x in planes]
    
    if max(radius) < 1:
        return clip
    
    mixproc = numplanes - len(planes) > 1
    mixproc = mixproc or any(len(set(p))>1 for p in (pr, pm, pb))
    
    if mixproc:
        clips = split(clip)
        return join([sbr_internal(clips[x], radius[x], [3], mode[x], blur[x]) for x in range(numplanes)])
    
    return sbr_internal(clip, pr[0], aplanes, pm[0], pb[0])

sbrV = partial(sbr, mode='v')
sbrH = partial(sbr, mode='h')

def sbr_internal(clip, radius, aplanes, mode, blur):
    
    if radius < 1:
        return clip
    
    vplanes = avs_to_vs(aplanes)
    
    return MinFilter(clip, partial(Blur, radius=radius, planes=vplanes, mode=mode, blur=blur), planes=vplanes)





def MinFilter(clip, filtera, filterb=None, mode=None, planes=None):
    mode = fallback(mode, 2 if filterb is None else 1)
    filterb = fallback(filterb, filtera)
    if mode==1:
        return MedianClip([clip, filtera(clip), filterb(clip)], planes=planes)
    core = vs.core
    filtered = filtera(clip)
    diffa = core.std.MakeDiff(clip, filtered, planes=planes)
    diffb = filterb(diffa)
    return MedianDiff(clip, diffa, diffb, planes=planes)



def MedianClip(clips, planes=None):
    core = vs.core
    numplanes = clips[0].format.num_planes
    planes = parse_planes(planes, numplanes, 'MedianClip')
    return core.std.Expr(clips, ['x y z min max y z max min' if x in planes else '' for x in range(numplanes)])



def MedianDiff(clip, diffa, diffb, planes=None):
    core = vs.core
    fmt = clip.format
    numplanes = fmt.num_planes
    planes = parse_planes(planes, numplanes, 'MedianDiff')
    
    neutral = f' {1 << (fmt.bits_per_sample - 1)} - ' if fmt.sample_type==vs.INTEGER else ''
    expr = f'x 0 y {neutral} y z - min max y {neutral} y z - max min -'
    
    return core.std.Expr([clip, diffa, diffb], [expr if x in planes else '' for x in range(numplanes)])





removegrain = RemoveGrain
removegrainm = RemoveGrainM

repair = Repair

median = Median

blur = Blur

minblur = MinBlur
minblurv = MinBlurV
minblurh = MinBlurH

sbrv = sbrV
sbrh = sbrH





def Depth(clip, planes, range_in, cal, outfmt=None):
    core = vs.core
    fmt = clip.format
    numplanes = fmt.num_planes
    range_in = 1 if range_in else 0
    yexpr = ['x 56064 * 4096 +', 'x 65535 *'][range_in] if outfmt is None else ['x 4096 - 56064 /', 'x 65535 /'][range_in]
    cexpr = ['x 57344 * 32768 +', 'x 65535 * 32768 +'][range_in] if outfmt is None else ['x 32768 - 57344 /', 'x 32768 - 65535 /'][range_in]
    expr = [yexpr] * 3 if fmt.color_family == vs.RGB else [yexpr, cexpr, cexpr]
    outfmt = fallback(outfmt, clip.format.replace(bits_per_sample=16, sample_type=vs.INTEGER))
    if cal:
        return core.std.Expr(clip, expr[-1], outfmt)
    if len(planes) != numplanes:
        expr = [expr[x] if x in planes else '' for x in range(numplanes)]
        return core.std.Expr(clip, expr, outfmt.id)
    return core.resize.Point(clip, range=range_in, range_in=range_in, format=outfmt.id)

def rep1(clip, repairclip, planes):
    core = vs.core
    if len(planes)==0:
        return clip
    rmax = repairclip.std.Maximum(planes=planes)
    rmin = repairclip.std.Minimum(planes=planes)
    mexpr = ['x y min z max' if i in planes else '' for i in range(clip.format.num_planes)]
    return core.std.Expr([clip, rmax, rmin], mexpr)

def rg4(clip, planes):
    if len(planes)==0:
        return clip
    core = vs.core
    return core.std.Median(clip, planes=planes)

def rg11(clip, planes):
    if len(planes)==0:
        return clip
    core = vs.core
    return core.std.Convolution(clip, [1,2,1,2,4,2,1,2,1], planes=planes)

def rg19(clip, planes):
    if len(planes)==0:         
        return clip
    core = vs.core
    return core.std.Convolution(clip, [1,1,1,1,0,1,1,1,1], planes=planes)

def rg20(clip, planes):
    if len(planes)==0:
        return clip
    core = vs.core
    return core.std.Convolution(clip, [1]*9, planes=planes)

def get_namespace(clip): 
    core = vs.core
    return core.rgsf if clip.format.sample_type==vs.FLOAT else core.rgvs

def avs_to_vs(planes):
    out = []
    for x in range(len(planes)):
        if planes[x] == 3:
            out += [x]
    return out

def vs_to_avs(planes, numplanes=3):
    out = [3 if x in planes else 1 for x in range(numplanes)]
    return out

def parse_planes(planes, numplanes, name):
    out = list(range(numplanes)) if planes is None else [planes] if isinstance(planes, int) else planes
    out = out[:min(len(out), numplanes)]
    for x in out:
        if x >= numplanes:
            raise TypeError(f'{name}: one or more "planes" values out of bounds')
    return out

def append_params(params, length=3):
    if not isinstance(params, list):
        params = [params]
    while len(params)<length:
        params.append(params[-1])
    return params
