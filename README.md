# RgToolsVS
Wrapper for RGVS, RGSF, and various median plugins as well as some functions that largely utilize them

#### Optimizations
- Replaces certain depreciated modes in RGVS/RGSF with core plugins
- Functions `sbr` and `MinBlur` have been completely rewritten to speed them up as much as possible

#### Enhancements
- Add `planes` parameters to functions and plugins that previously didn't have them. Takes priority over "mode" and "radius", setting them to 0
- Allow `radius` parameter to be set per-plane (sbr and MinBlur)
- Support float input (CTMF, sbr and MinBlur)

#### "New" Functions
- Blur - Wrapper for std.BoxBlur, std.Convolution and iterative RemoveGrain usage with mode 11, 20, 20, etc
- Sharpen - Port of the Avisynth internal plugin with adjustable radius
- RemoveGrainM - Helper function for iterative RemoveGrain

#### Dependencies
- [vsutil](https://github.com/Irrational-Encoding-Wizardry/vsutil/blob/master/py)
- [RGSF](https://github.com/IFeelBloated/RGSF)
- [CTMF](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-CTMF)
- [fmtconv](https://github.com/EleonoreMizo/fmtconv/)
- [TemporalMedian](https://github.com/dubhater/vapoursynth-temporalmedian)
- [vs-average](https://github.com/End-of-Eternity/vs-average)

## RGVS/RGSF
#### All
- Internally detect sample type and choose `core.rgvs` or `core.rgsf` automatically (this is the main reason why I wrote this thing in the first place)

#### Repair
- Use std.Minimum, std.Maximum and std.Expr to perform mode 1 in RGSF (commentable)

#### RemoveGrain
- Use std.Convolution and std.Median for modes 4, 11, 12, 19 and 20 (commentable)

#### Clense
- Use std.Expr instead of RGVS (commentable)

See Vapoursynth website for more information on RGVS - http://www.vapoursynth.com/doc/plugins/rgvs.html


## Median 
Wraps std.Median, ctmf.CTMF, rgvs.VerticalMedian and average.Median into one function
```
Median(clip clip[, int[] radius=[1, 1, 1], int[] planes=[0, 1, 2], str[] mode=['s', 's', 's'], int[] vcmode=[1, 1, 1], bint range_in=color_family==vs.RGB, int memsize=1048576, int opt=0])
```
- "vcmode" allows you to change the VerticalMedian mode when using horizontal or vertical processing and a radius of 1
- "range_in" is for mixed float/int processing when using CTMF
- "memsize" and "opt" parameters from CTMF


## RemoveGrainM
Special helper function for iterative RemoveGrain usage
```
RemoveGrainM(clip clip[, int[] mode=[2, 2, 2], int[] modeu=mode, int[] modev=modeu, int[] iter=max length of modes, int[] planes=[0, 1, 2] ])
```
```
RemoveGrainM(clip, mode=1, modeu=2, modev=3, iter=[1,2,3])
# is equal to
clip.rgvs.RemoveGrain([1,2,3]).rgvs.RemoveGrain([0,2,3]).rgvs.RemoveGrain([0,0,3])
```


## More functions
### MinBlur
Pixelwise median and blur admixture. If the differences are homologous to the input, the weaker result of the two are taken, else the source pixel is passed
```
MinBlur(clip clip[, int[] radius=[1, 1, 1], int[] planes=[0, 1, 2], str[] mode=['s', 's', 's'], str[] bmode=['gauss', 'gauss', 'gauss'], int[] vcmode=[1, 1, 1], bint range_in=color_family==vs.RGB, int memsize=1048576, int opt=0])
```
### sbr
Pixelwise blur operation. Takes a blur's difference and performs an equal blurring upon said difference. If both differences are homologous to the input, the weaker difference is reapplied, else the source pixel is passed
```
sbr(clip clip[, int[] radius=[1, 1, 1], int[] planes=[0, 1, 2], str[] mode=['s', 's', 's'], str[] bmode=['gauss', 'gauss', 'gauss'] ])
```
### Blur
No relation to AviSynth's internal plugin, use Sharpen with negative values if you want that.
Like RemoveGrainM, this was mainly written as a helper for other functions.
```
Blur(clip clip[, int[] radius=[1, 1, 1], int[] planes=[0, 1, 2], str[] mode=['s', 's', 's'], str[] bmode=['gauss', 'gauss', 'gauss'] ])
```
- Included modes:
  - "box" - average blurring (same as RemoveGrain(20), Convolution([1]\*9) and BoxBlur)
  - "gauss" - pseudo-gaussian blur technique made popular on a certain forum using an iteration of removegrain, starting with a call to mode 11 and all subsequent calls to mode 20
- parameter "radius" is iterative when mode="gauss"
- parameter "bmode" toggles between std.BoxBlur for bmode="box" and removegrain(11)\[.removegrain(20)...] for bmode="gauss"

### Sharpen
Port of the AviSynth internal plugin. Please use 1.585 as opposed to 1.58 for box blurring in integer precision. For float precision `import math` and use `math.log2(3)`
```
Sharpen(clip clip[, int[] amountH=1, int[] amountV=amountH, int[] radius=1, int[] planes=None, bint[] legacy=False])
```
Accepts `amount` as `amountH` alias

Radius can be up to `2` for 2D processing (3x3, 5x5) and up to `12` for horizontal or vertical-only processing.

Set legacy to clamp `amount` to the range `-158` to `1`
