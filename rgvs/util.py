__all__ = ['MinFilter', 'MedianClip', 'MedianDiff']

import vapoursynth as vs
from vsutil import split, join, fallback





def MinFilter(clip, filtera, filterb=None, mode=None, planes=None):
    if mode == 1 and filterb is None:
        raise ValueError('rgvs.MinFilter: filterb must be defined when mode=1')
    
    planes = parse_planes(planes, clip.format.num_planes, 'MinFilter')
    mode = fallback(mode, 2 if filterb is None else 1)
    
    filterb = fallback(filterb, filtera)
    
    if mode == 1:
        return MedianClip([clip, filtera(clip), filterb(clip)], planes=planes)
        
    filtered = filtera(clip)
    
    diffa = clip.std.MakeDiff(filtered, planes=planes)
    diffb = filterb(diffa)
    
    return MedianDiff(clip, diffa, diffb, planes=planes)



def MedianClip(clips, planes=None):
    core = vs.core
    
    numplanes = clips[0].format.num_planes
    planes = parse_planes(planes, numplanes, 'MedianClip')
    
    if rad <= 10:
        dia = len(clips)
        rad = dia // 2
        return core.std.Interleave(clips).tmedian.TemporalMedian(radius=rad, planes=planes)[rad::dia]
    
    return core.average.Median(clips)



def MedianDiff(clip, diffa, diffb, planes=None):
    core = vs.core
    
    fmt = clip.format
    numplanes = fmt.num_planes
    planes = parse_planes(planes, numplanes, 'MedianDiff')
    
    neutral = f' {1 << (fmt.bits_per_sample - 1)} - ' if fmt.sample_type==vs.INTEGER else ''
    expr = f'x 0 y {neutral} y z - min max y {neutral} y z - max min -'
    
    return core.std.Expr([clip, diffa, diffb], [expr if x in planes else '' for x in range(numplanes)])



def parse_planes(planes, numplanes, name):
    planes = fallback(planes, list(range(numplanes)))
    if not isinstance(planes, (list, tuple, set, int)):
        raise TypeError(f'{name}: improper "planes" format')
    if isinstance(planes, int):
        planes = [planes]
    else:
        planes = list(set(planes))
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
