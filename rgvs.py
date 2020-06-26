import vapoursynth as vs
from vsutil import split, join, fallback
from functools import partial
from math import ceil, log2





def Repair(clip, repairclip, mode=2, planes=None):
    if not isinstance(clip, vs.VideoNode):
        raise TypeError('Repair: "clip" is not a clip')
    if not isinstance(repairclip, vs.VideoNode):
        raise TypeError('Repair: "repairclip" is not a clip')
    
    fmt = clip.format
    numplanes = fmt.num_planes
    
    planes = parse_planes(planes, numplanes, 'Repair')
    
    mode = append_params(mode, numplanes)
    mode = [max(mode[x], 0) if x in planes else 0 for x in range(numplanes)]
    
################################################
### Mode 1 - Use Minimum/Maximum for float clips
    if fmt != repairclip.format:
        raise vs.Error('Repair: "clip" and "repairclip" must have matching formats')
    planes = []
    if fmt.sample_type == vs.FLOAT:
        for i in range(numplanes):
            if mode[i] in (1, 11):
                mode[i] = 0
                planes = planes + [i]
    
    clip = rep1(clip, repairclip, planes)
################################################
    
    if max(mode) == 0:
        return clip
    return get_namespace(clip).Repair(clip, repairclip, mode)



def RemoveGrain(clip, mode=2, planes=None):
    if not isinstance(clip, vs.VideoNode):
        raise TypeError('RemoveGrain: This is not a clip')
    
    fmt = clip.format
    numplanes = fmt.num_planes
    isint = fmt.sample_type == vs.INTEGER
    
    mode = append_params(mode, numplanes)
    
    planes = parse_planes(planes, numplanes, 'RemoveGrain')
    mode = [max(mode[x], 0) if x in planes else 0 for x in range(numplanes)]
    
    planes_rg4  = []
    planes_rg11 = []
    planes_rg19 = []
    planes_rg20 = []
    
#########################################################
### Mode 4 - Use std.Median() ###########################
    for i in range(numplanes):
        if mode[i] == 4:
            planes_rg4 += [i]
            mode[i] = 0
    clip = rg4 (clip, planes_rg4)
#########################################################

#########################################################
### Mode 11/12 - Use std.Convolution([1,2,1,2,4,2,1,2,1])
    for i in range(numplanes):
        if mode[i] in (11, 12):
            planes_rg11 += [i]
            mode[i] = 0
    clip = rg11(clip, planes_rg11)
#########################################################

########################################################
### Mode 19 - Use std.Convolution([1,1,1,1,0,1,1,1,1])
    for i in range(numplanes):
        if mode[i] == 19:
            planes_rg19 += [i]
            mode[i] = 0
    clip = rg19(clip, planes_rg19)
########################################################

########################################################
### Mode 20 - Use std.Convolution([1,1,1,1,1,1,1,1,1])
    for i in range(numplanes):
        if mode[i] == 20:
            planes_rg20 += [i]
            mode[i] = 0
    clip = rg20(clip, planes_rg20)
########################################################
    
    if max(mode) == 0:
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
    
    planes = parse_planes(planes, clip.format.num_planes, 'Clense')
    if grey:
        planes = eval_planes(planes, [1] + [0] * (numplanes - 1), 0)
    if len(planes) == 0:
        return clip
    
###################################################################
### Use std.Expr for integer and half float clips #################
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
    planes = parse_planes(planes, clip.format.num_planes, 'ForwardClense')
    if grey:
        planes = eval_planes(planes, [1] + [0] * (numplanes - 1), 0)
    if len(planes) == 0:
        return clip
    return get_namespace(clip).ForwardClense(clip, planes)



def BackwardClense(clip, planes=None, grey=False):
    if not isinstance(clip, vs.VideoNode):
        raise TypeError('BackwardClense: This is not a clip')
    planes = parse_planes(planes, clip.format.num_planes, 'BackwardClense')
    if grey:
        planes = eval_planes(planes, [1] + [0] * (numplanes - 1), 0)
    if len(planes) == 0:
        return clip
    return get_namespace(clip).BackwardClense(clip, planes)



def VerticalCleaner(clip, mode=1, planes=None):
    if not isinstance(clip, vs.VideoNode):
        raise TypeError('VerticalCleaner: This is not a clip')
    
    numplanes = clip.format.num_planes
    
    mode = append_params(mode, numplanes)
    
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
    
    mode  = [0] if iter[0] is 0 else append_params(mode , iter[0])
    modeu = [0] if iter[1] is 0 else append_params(modeu, iter[1])
    modev = [0] if iter[2] is 0 else append_params(modev, iter[2])
    
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
    
    radius = fallback(radius, r)
    
    radius = append_params(radius, numplanes)
    mode   = append_params(mode,   numplanes)
    vcmode = append_params(vcmode, numplanes)
    
    planes = parse_planes(planes, numplanes, 'Median')
    planes = eval_planes(planes, radius, zero=0)
    
    if len(planes) == 0:
        return clip
    
    pr = [radius[x] for x in planes]
    pm = [mode[x]   for x in planes]
    pv = [vcmode[x] for x in planes]
    
    mixproc = ('h' in pm or 'v' in pm) and max(pr) > 1 and numplanes - len(planes) != 0 # no planes parameter in average.Median
    mixproc = mixproc or any(len(set(p))>1 for p in (pr, pm, pv))
    
    if mixproc:
        clips = split(clip)
        for x in planes:
            clip[x] = Median_internal(clips[x], radius[x], [3], mode[x], vcmode[x], range_in, [0,1,1][x], memsize, opt)
        return join(clips)
    
    return Median_internal(clip, pr[0], vs_to_avs(planes, numplanes), pm[0], pv[0], range_in, 0, memsize, opt)

def Median_internal(clip, radius, aplanes, mode, vcmode, range_in, cal, memsize, opt):
    core = vs.core
    
    fmt = clip.format
    numplanes = fmt.num_planes
    
    mode = mode.lower()
    
    vplanes = avs_to_vs(aplanes)
    
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
            clips[x] = Blur_internal(clips[x], radius[x], [3], mode[x], bmode[x])
        return join(clips)
    
    return Blur_internal(clip, pr[0], vs_to_avs(planes, numplanes), pm[0], pb[0])

def Blur_internal(clip, radius, aplanes, mode, bmode):
    core = vs.core
    
    fmt = clip.format
    isint = fmt.sample_type == vs.INTEGER
    numplanes = fmt.num_planes
    
    bmode = bmode.lower()
    mode = mode.lower()
    
    vplanes = avs_to_vs(aplanes)
    
    if 'g' not in bmode:
        if radius==1:
            if mode=='s':
                return RemoveGrain(clip, mode=[20,20,20], planes=vplanes)
            return core.std.Convolution(clip, matrix=[1,1,1], planes=vplanes, mode=mode)
        return core.std.BoxBlur(clip, planes=vplanes, hradius=0 if mode=='v' else radius, hpasses=1, vradius=0 if mode=='h' else radius, vpasses=1)
    
    if mode == 's':
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
    
    core = vs.core
    
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
    
    if not 0 < radius < (3 if mode is 's' else 13)
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
    
    return core.std.Convolution(clip, matrix=matrix, planes=planes, mode=mode)



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
    
    if max(pr) < 1:
        return clip
    
    mixproc = numplanes - len(planes) > 1
    mixproc = mixproc or any(len(set(p))>1 for p in (pr, pm, pb))
    
    if mixproc:
        clips = split(clip)
        return join([sbr_internal(clips[x], radius[x], [3], mode[x], bmode[x]) for x in range(numplanes)])
    
    return sbr_internal(clip, pr[0], vs_to_avs(planes, numplanes), pm[0], pb[0])

sbrV = partial(sbr, mode='v')
sbrH = partial(sbr, mode='h')

def sbr_internal(clip, radius, aplanes, mode, bmode):
    
    if radius < 1:
        return clip
    
    vplanes = avs_to_vs(aplanes)
    
    return MinFilter(clip, partial(Blur, radius=radius, planes=vplanes, mode=mode, bmode=bmode), planes=vplanes)





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
sharpen = Sharpen

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
    return [core.rgvs, core.rgsf][clip.format.sample_type]

def avs_to_vs(planes):
    out = []
    for x in range(len(planes)):
        if planes[x] == 3:
            out.append(x)
    return out

def vs_to_avs(planes, numplanes=3): return [3 if x in planes else 1 for x in range(numplanes)]

def parse_planes(planes, numplanes, name):
    planes = fallback(planes, list(range(numplanes)))
    if isinstance(planes, int):
        planes = [planes]
    if isinstance(planes, tuple):
        planes = list(planes)
    if not isinstance(planes, list):
        raise TypeError(f'{name}: improper "planes" format')
    planes = planes[:min(len(planes), numplanes)]
    for x in planes:
        if x >= numplanes:
            planes = planes[:planes.index(x)]
            break
    return planes

# if param[x] is a number less than or equal to "zero" or is explicitly False or None, delete x from "planes"
# if param[x] is a number greater than "zero" or is explicitly True, pass x if it was originally in "planes"
def eval_planes(planes, params, zero=0):
    process = []
    for x in range(len(params)):
        if x in planes:
            if params[x] is False or params[x] is None:
                pass
            elif params[x] is True:
                process += [x]
            elif params[x] > zero:
                process += [x]
    return process

def append_params(params, length=3):
    if isinstance(params, tuple):
        params = list(params)
    if not isinstance(params, list):
        params = [params]
    while len(params) < length:
        params.append(params[-1])
    return params[:length]

def get_matrix_array(rwh, rwv, radius, center, mode):
    matrix = [center]
    for x in range(radius):
        matrix.append(abs(matrix[-1]) / rwh)
    if mode == 's':
        for x in range(radius):
            for y in range(radius + 1):
                matrix.append(abs(matrix[(radius + 1) * x + y]) / rwv)
    return matrix
