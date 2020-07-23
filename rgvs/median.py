__all__ = ['Median']

import vapoursynth as vs
from .util import split, join, fallback, append_params, parse_planes, eval_planes, MedianClip





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
    
    mixproc = any(len(set(p))>1 for p in (pr, pm, pv))
    
    if mixproc:
        clips = split(clip)
        for x in planes:
            clip[x] = Median_internal(clips[x], radius[x], [0], mode[x], vcmode[x], range_in, [0,1,1][x], memsize, opt)
        return join(clips)
    
    return Median_internal(clip, pr[0], planes, pm[0], pv[0], range_in, 0, memsize, opt)



def Median_internal(clip, radius, planes, mode, vcmode, range_in, cal, memsize, opt):
    
    fmt = clip.format
    numplanes = fmt.num_planes
    
    mode = mode.lower()
    
    if mode == 's':
        if radius == 1:
            return clip.std.Median(clip, planes=planes)
        
        if fmt.sample_type == vs.FLOAT:
            core = vs.core
            down = _depth(clip, planes, range_in, cal)
            medd = core.ctmf.CTMF(down, radius=radius, memsize=memsize, opt=opt, planes=planes)
            medu = _depth(medd, planes, range_in, cal, fmt)
            return core.std.Expr([clip, medu, down, medd], ['z a = x y ?' if x in planes else '' for x in range(numplanes)])
        
        return clip.ctmf.CTMF(radius=radius, memsize=memsize, opt=opt, planes=planes)
    
    return Median2D(clip, radius, mode, planes)



def Median2D(clip, radius, mode, planes):
    
    if radius > 10 and len(planes) > clip.format.num_planes:
        clips = split(clip)
        for x in clips:
            clips[x] = Median2D(clips[x], radius, mode, [0])
        return join(clips)
    
    if radius == 1:
        if mode == 'v':
            return VerticalCleaner(clip, mode=vcmode, planes=vplanes)
        return VerticalCleaner(clip.std.Transpose(), mode=vcmode, planes=vplanes).std.Transpose()
    
    ShiftPlanes = partial(_shift, sx=0) if mode == 'v' else partial(_shift, sy=0)
    
    clips = [ShiftPlanes(clip, x, planes) for x in range(-radius, radius + 1)]
    
    return MedianClip(clips, planes)



def _shift(clip, sx, sy, planes):
    if sx == sy == 0:
        return clip
    
    fmt = clip.format
    bits = fmt.bits_per_sample
    hss = fmt.subsampling_w
    vss = fmt.subsampling_h
    numplanes = fmt.num_planes
    
    if bits > 8:
        sx = append_params([sx, sx << hss], numplanes)
        sy = append_params([sy, sy << vss], numplanes)
        planes = [3 if x in planes else 1 for x in range(numplanes)]
        return clip.fmtc.resample(sx=sx, sy=sy, kernel='point', planes=planes)
    
    clips = split(clip)
    
    for x in planes:
        clips[x] = clips[x].resize.Point(src_left=sx, src_top=sy)
    
    return join(clips)



def _depth(clip, planes, range_in, cal, outfmt=None):
    fmt = clip.format
    numplanes = fmt.num_planes
    range_in = 1 if range_in else 0
    
    yexpr = ['x 56064 * 4096 +', 'x 65535 *'][range_in] if outfmt is None else ['x 4096 - 56064 /', 'x 65535 /'][range_in]
    cexpr = ['x 57344 * 32768 +', 'x 65535 * 32768 +'][range_in] if outfmt is None else ['x 32768 - 57344 /', 'x 32768 - 65535 /'][range_in]
    
    expr = [yexpr] * 3 if fmt.color_family == vs.RGB else [yexpr, cexpr, cexpr]
    
    outfmt = fallback(outfmt, clip.format.replace(bits_per_sample=16, sample_type=vs.INTEGER))
    
    if cal:
        return clip.std.Expr(expr[-1], outfmt)
    
    f16_out = outfmt.bits_per_sample == 16 and outfmt.sample_type == vs.FLOAT
    f16_in = fmt.bits_per_sample == 16 and fmt.sample_type == vs.FLOAT
    
    if len(planes) != numplanes:
        if f16_in or f16_out:
            return clip.std.Expr(expr, outfmt.id)
        else:
            return clip.fmtc.bitdepth(csp=outfmt.id, planes=planes, fulls=range_in, fulld=range_in, dmode=1)
    return clip.resize.Point(range=range_in, range_in=range_in, format=outfmt.id)
