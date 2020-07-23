__all__ = ['Repair', 'RemoveGrainM', 'RemoveGrain', 'VerticalCleaner', 'Clense', 'BackwardClense', 'ForwardClense']

import vapoursynth as vs
from .util import fallback, append_params, parse_planes, eval_planes, MedianClip





def Repair(clip, repairclip, mode=2, planes=None):
    if not isinstance(clip, vs.VideoNode):
        raise TypeError('Repair: "clip" is not a clip')
    if not isinstance(repairclip, vs.VideoNode):
        raise TypeError('Repair: "repairclip" is not a clip')
    
    fmt = clip.format
    isflt = fmt.sample_type
    numplanes = fmt.num_planes
    
    planes = parse_planes(planes, numplanes, 'Repair')
    
    mode = append_params(mode, numplanes)
    mode = [max(mode[x], 0) if x in planes else 0 for x in range(numplanes)]
    
################################################################################################
################################################################################################
### Mode 1 - Use Minimum/Maximum for float clips
    if fmt != repairclip.format:
        raise vs.Error('Repair: "clip" and "repairclip" must have matching formats')
    def _rep1(clip, repairclip, planes):
        if len(planes)==0:
            return clip
        core = vs.core
        rmax = repairclip.std.Maximum(planes=planes)
        rmin = repairclip.std.Minimum(planes=planes)
        expr = ['x y min z max' if i in planes else '' for i in range(clip.format.num_planes)]
        return core.std.Expr([clip, rmax, rmin], expr)
    planes = []
    if fmt.sample_type == vs.FLOAT:
        for i in range(numplanes):
            if mode[i] in (1, 11):
                mode[i] = 0
                planes = planes + [i]
    clip = _rep1(clip, repairclip, planes)
################################################################################################
################################################################################################
    
    if max(mode) == 0:
        return clip
    if isflt:
        return clip.rgsf.Repair(repairclip, mode)
    return clip.rgvs.Repair(repairclip, mode)



def RemoveGrain(clip, mode=2, planes=None):
    if not isinstance(clip, vs.VideoNode):
        raise TypeError('RemoveGrain: This is not a clip')
    
    fmt = clip.format
    isflt = fmt.sample_type
    numplanes = fmt.num_planes
    
    mode = append_params(mode, numplanes)
    
    planes = parse_planes(planes, numplanes, 'RemoveGrain')
    mode = [max(mode[x], 0) if x in planes else 0 for x in range(numplanes)]
    
########################################################################
########################################################################
### Mode 4 - Use std.Median()
    def _rg4(clip, planes):
        if len(planes)==0:
            return clip
        return clip.std.Median(planes=planes)
    planes_rg4  = []
    for i in range(numplanes):
        if mode[i] == 4:
            planes_rg4 += [i]
            mode[i] = 0
    clip = _rg4(clip, planes_rg4)
########################################################################
########################################################################


########################################################################
########################################################################
### Mode 11/12 - Use std.Convolution([1,2,1,2,4,2,1,2,1])
    def _rg11(clip, planes):
        if len(planes)==0:
            return clip
        return clip.std.Convolution([1,2,1,2,4,2,1,2,1], planes=planes)
    planes_rg11 = []
    for i in range(numplanes):
        if mode[i] in (11, 12):
            planes_rg11 += [i]
            mode[i] = 0
    clip = _rg11(clip, planes_rg11)
########################################################################
########################################################################


########################################################################
########################################################################
### Mode 19 - Use std.Convolution([1,1,1,1,0,1,1,1,1])
    def _rg19(clip, planes):
        if len(planes)==0:         
            return clip
        return clip.std.Convolution([1,1,1,1,0,1,1,1,1], planes=planes)
    planes_rg19 = []
    for i in range(numplanes):
        if mode[i] == 19:
            planes_rg19 += [i]
            mode[i] = 0
    clip = _rg19(clip, planes_rg19)
########################################################################
########################################################################


########################################################################
########################################################################
### Mode 20 - Use std.Convolution([1,1,1,1,1,1,1,1,1])
    def _rg20(clip, planes):
        if len(planes)==0:
            return clip
        return clip.std.Convolution([1]*9, planes=planes)
    planes_rg20 = []
    for i in range(numplanes):
        if mode[i] == 20:
            planes_rg20 += [i]
            mode[i] = 0
    clip = _rg20(clip, planes_rg20)
########################################################################
########################################################################
    
    if max(mode) == 0:
        return clip
    if isflt:
        return clip.rgsf.RemoveGrain(mode)
    return clip.rgvs.RemoveGrain(mode)



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
    
    if previous == next == None:
        return clip.tmedian.TemporalMedian(1, planes)
    
    previous = fallback(previous, clip)
    previous = previous[0] + previous[:-1]
    
    next = fallback(next, clip)
    next = next[1:] + next[-1]
    
    return MedianClip([previous, clip, next], planes=planes)



def ForwardClense(clip, planes=None, grey=False):
    if not isinstance(clip, vs.VideoNode):
        raise TypeError('ForwardClense: This is not a clip')
    
    fmt = clip.format
    isflt = fmt.sample_type
    numplanes = fmt.num_planes
    
    planes = parse_planes(planes, clip.format.num_planes, 'ForwardClense')
    
    if grey:
        planes = eval_planes(planes, [1] + [0] * (numplanes - 1), 0)
    
    if len(planes) == 0:
        return clip
    if isflt:
        return clip.rgsf.ForwardClense(planes)
    return clip.rgvs.ForwardClense(planes)



def BackwardClense(clip, planes=None, grey=False):
    if not isinstance(clip, vs.VideoNode):
        raise TypeError('BackwardClense: This is not a clip')
    
    fmt = clip.format
    isflt = fmt.sample_type
    numplanes = fmt.num_planes
    
    planes = parse_planes(planes, clip.format.num_planes, 'BackwardClense')
    
    if grey:
        planes = eval_planes(planes, [1] + [0] * (numplanes - 1), 0)
    
    if len(planes) == 0:
        return clip
    if isflt:
        return clip.rgsf.BackwardClense(planes)
    return clip.rgvs.BackwardClense(planes)



def VerticalCleaner(clip, mode=1, planes=None):
    if not isinstance(clip, vs.VideoNode):
        raise TypeError('VerticalCleaner: This is not a clip')
    
    fmt = clip.format
    isflt = fmt.sample_type
    numplanes = fmt.num_planes
    
    mode = append_params(mode, numplanes)
    
    planes = parse_planes(planes, numplanes, 'VerticalCleaner')
    mode = [max(mode[x], 0) if x in planes else 0 for x in range(numplanes)]
    
    if max(mode) == 0:
        return clip
    if isflt:
        return clip.rgsf.VerticalCleaner(mode)
    return clip.rgvs.VerticalCleaner(mode)
