__all__ = ['repair', 'removegrainm', 'removegrain', 'verticalcleaner', 'clense', 'backwardclense', 'forwardclense']

import vapoursynth as vs

from .util import medianclip, append_params, eval_planes, fallback, parse_planes


def repair(clip, repairclip, mode=2, planes=None):
    if not isinstance(clip, vs.VideoNode):
        raise TypeError('repair: "clip" is not a clip')
    if not isinstance(repairclip, vs.VideoNode):
        raise TypeError('repair: "repairclip" is not a clip')

    fmt = clip.format
    isflt = fmt.sample_type
    numplanes = fmt.num_planes

    planes = parse_planes(planes, numplanes, 'repair')

    mode = append_params(mode, numplanes)
    mode = [max(mode[x], 0) if x in planes else 0 for x in range(numplanes)]

################################################################################################
################################################################################################
### Mode 1 - Use Minimum/Maximum for float clips
    if fmt != repairclip.format:
        raise vs.Error('repair: "clip" and "repairclip" must have matching formats')
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
        return clip.rgsf.repair(repairclip, mode)
    return clip.rgvs.repair(repairclip, mode)


def removegrain(clip, mode=2, planes=None):
    if not isinstance(clip, vs.VideoNode):
        raise TypeError('removegrain: This is not a clip')

    fmt = clip.format
    isflt = fmt.sample_type
    numplanes = fmt.num_planes

    mode = append_params(mode, numplanes)

    planes = parse_planes(planes, numplanes, 'removegrain')
    mode = [max(mode[x], 0) if x in planes else 0 for x in range(numplanes)]

########################################################################
########################################################################
### Mode 4 - Use std.median()
    def _rg4(clip, planes):
        if len(planes)==0:
            return clip
        return clip.std.median(planes=planes)
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
        return clip.rgsf.removegrain(mode)
    return clip.rgvs.removegrain(mode)


def removegrainm(clip, mode=2, modeu=None, modev=None, iter=None, planes=None):
    if not isinstance(clip, vs.VideoNode):
        raise TypeError('removegrainm: This is not a clip')

    numplanes = clip.format.num_planes
    planes = parse_planes(planes, numplanes, 'removegrainm')

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
        clip = removegrain(clip, [mode[x], modeu[x], modev[x]])

    return clip


def clense(clip, previous=None, next=None, planes=None, grey=False):
    if not isinstance(clip, vs.VideoNode):
        raise TypeError('clense: This is not a clip')
    if next is not None:
        if not isinstance(next, vs.VideoNode):
            raise TypeError('clense: This is not a clip')
    if previous is not None:
        if not isinstance(previous, vs.VideoNode):
            raise TypeError('clense: This is not a clip')

    planes = parse_planes(planes, clip.format.num_planes, 'clense')
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

    return medianclip([previous, clip, next], planes=planes)


def forwardclense(clip, planes=None, grey=False):
    if not isinstance(clip, vs.VideoNode):
        raise TypeError('forwardclense: This is not a clip')

    fmt = clip.format
    isflt = fmt.sample_type
    numplanes = fmt.num_planes

    planes = parse_planes(planes, clip.format.num_planes, 'forwardclense')

    if grey:
        planes = eval_planes(planes, [1] + [0] * (numplanes - 1), 0)

    if len(planes) == 0:
        return clip
    if isflt:
        return clip.rgsf.forwardclense(planes)
    return clip.rgvs.forwardclense(planes)


def backwardclense(clip, planes=None, grey=False):
    if not isinstance(clip, vs.VideoNode):
        raise TypeError('backwardclense: This is not a clip')

    fmt = clip.format
    isflt = fmt.sample_type
    numplanes = fmt.num_planes

    planes = parse_planes(planes, clip.format.num_planes, 'backwardclense')

    if grey:
        planes = eval_planes(planes, [1] + [0] * (numplanes - 1), 0)

    if len(planes) == 0:
        return clip
    if isflt:
        return clip.rgsf.backwardclense(planes)
    return clip.rgvs.backwardclense(planes)


def verticalcleaner(clip, mode=1, planes=None):
    if not isinstance(clip, vs.VideoNode):
        raise TypeError('verticalcleaner: This is not a clip')

    fmt = clip.format
    isflt = fmt.sample_type
    numplanes = fmt.num_planes

    mode = append_params(mode, numplanes)

    planes = parse_planes(planes, numplanes, 'verticalcleaner')
    mode = [max(mode[x], 0) if x in planes else 0 for x in range(numplanes)]

    if max(mode) == 0:
        return clip
    if isflt:
        return clip.rgsf.verticalcleaner(mode)
    return clip.rgvs.verticalcleaner(mode)
