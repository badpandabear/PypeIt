# Fiber Superflat and 1D Sky Subtraction Design

**Date**: 2026-03-28
**Branch**: binospec_updates
**Status**: Approved for implementation
**Supersedes**: Sky subtraction sections of 2026-03-25-fiber-block-slit-extraction-design.md

## Problem

The block-slit extraction design (2026-03-25) introduced a sky subtraction
approach that caused three compounding problems:

1. **Noise model corruption**: Extracting sky fibers to 1D, building a B-spline
   sky model, and projecting back to 2D loses per-pixel noise information. The
   variance model no longer represents actual detector noise after the
   round-trip.

2. **Variance mismatch**: `variance_model()` uses the projected sky counts,
   which don't match actual detector counts (pre-throughput-correction), causing
   incorrect noise estimates for optimal extraction.

3. **Spectral illumination failure**: The inherited `joint_skysub` method
   applies `illum_profile_spectral_poly`, which fits per-slit spectral response
   corrections. This diverges catastrophically for block-slits because fibers
   within a block have discrete throughput differences that cannot be modeled by
   a smooth polynomial. The pipeline crashes with `TypeError: expected
   non-empty vector for x` after the spectral illumination scales diverge to
   +/-10^7 and corrupt the ivar with NaN.

## Prior Art

### Binospec IDL Pipeline
- Achieves Poisson-noise-limited sky subtraction
- Processing order: extract fibers to 1D -> flat field + illumination correction
  -> wavelength linearize -> sky subtract on equalized 1D spectra
- `fiber_illumination.fits` is a static per-fiber scalar correction applied
  after extraction, before sky subtraction
- Flat field retains calibration lamp spectral signature (not spectrally
  normalized)

### MaNGA DRP (Law et al. 2016, AJ 152 83)
- Also achieves Poisson-limited sky subtraction
- **Superflat**: combine all extracted flat spectra (normalized to median=1)
  into super-sampled composite, B-spline fit -> global spectral response
- **Fiberflat**: individual fiber flat / superflat -> per-fiber relative
  throughput as a function of wavelength, near unity, slowly varying
- Science spectra divided by both superflat and fiberflat before sky subtraction
- Sky model: super-sampled B-spline from dedicated sky fibers with **smoothed**
  inverse-variance weights (critical to avoid Poisson bias)
- Per-harness gray scale factors account for spatial sky variations

## Design

### Overview

Adopt a MaNGA-style 1D sky subtraction approach adapted for the Binospec
block-slit architecture:

1. A new `FiberFlatField` subclass handles fiber-specific flat calibration,
   producing a superflat (common spectral response), per-fiber fiberflats
   (relative throughput), and a gray throughput ratio between sky and science
   fiber types.

2. Sky subtraction in `FiberFindObjects` operates on extracted 1D spectra:
   extract all fibers, equalize, build B-spline sky model from sky fibers,
   subtract sky from each science fiber's 1D spectrum. Both the sky model
   and the subtraction are in 1D — no projection back to 2D. A 2D sky
   image may be constructed for diagnostics only.

3. The block-slit architecture is preserved for edge detection, wavelength
   calibration (42 blocks, ~17x speedup), and fiber identification. Only the
   flat field and sky subtraction are reworked.

### Component 1: FiberFlatField

**Class**: `FiberFlatField(FlatField)` in `pypeit/flatfield.py`

**Dispatch**: `IFUCalibrations.get_flats()` overridden to instantiate
`FiberFlatField` when `pypeline == 'Fiber'`.

**2D Pixelflat**:
- Pixel-to-pixel response corrections only (high-frequency spatial variations,
  bad pixels)
- No spectral normalization, no illumination profile fitting
- Applied to the 2D science image via `apply_flat_fielding()` as usual

**1D Superflat + Fiberflat Construction**:

1. Extract all flat field fibers to 1D within their block-slits using boxcar
   extraction. Traces from fiber identification, aperture widths from
   inter-fiber spacing. Variance propagated from raw flat image.

2. Classify fibers by type (sky vs science) using
   `spectrograph.get_fiber_metadata()`. Sky fibers identified by `FIB_NAME`
   starting with `'SKY'`.

3. Build the **superflat** (single, from all fibers):
   - Compute each fiber's median flux -> per-fiber scale factor
   - Normalize each fiber spectrum to median=1
   - Combine all normalized spectra into a super-sampled 1D array sorted by
     wavelength (different fibers' wavelength solutions provide oversampling)
   - B-spline fit to the composite -> superflat (common spectral response
     shape, capturing lamp spectrum x system throughput)

4. Compute **gray throughput ratio**:
   - `sky_scale = median(scale_factors for sky fibers)`
   - `sci_scale = median(scale_factors for science fibers)`
   - `throughput_ratio = sky_scale / sci_scale` (single scalar)
   - This captures the geometric aperture difference between sky and science
     fiber types. Replaces `fiber_illumination.fits`.

5. Compute **fiberflat** per fiber:
   - `fiberflat_i(lambda) = fiber_spectrum_i(lambda) / (scale_factor_i * superflat(lambda))`
   - B-spline fit to smooth out photon noise and interpolate over bad pixels
   - Near unity, slowly varying with wavelength
   - Captures per-fiber throughput variation

**Output**: Extended `FlatImages` containing:
- `pixelflat_norm` -- 2D pixel-to-pixel correction (standard)
- `superflat` -- B-spline object (common spectral response)
- `fiberflat` -- 2D array (nfiber x nwave) of per-fiber throughput corrections
- `fiber_scale_factors` -- 1D array of per-fiber median scale factors
- `throughput_ratio` -- scalar (sky/science gray throughput)
- `fiber_ids` -- mapping from fiber index to fiber ID
- `fiber_types` -- sky vs science classification per fiber

### Component 2: Sky Subtraction in FiberFindObjects

**Replaces**: `global_skysub()` / `joint_skysub()` call chain. No spectral
illumination correction, no 2D sky model projection as primary step.

**Input**: Pixel-flat-corrected 2D science image (pixel-level clean, retains
lamp spectrum and fiber throughput differences).

**Algorithm**:

1. **Load calibration products**: superflat, fiberflat, scale factors,
   throughput ratio from `FiberFlatField`.

2. **Extract all fibers** (sky + science) from the 2D image via boxcar
   extraction within their block-slits. Variance propagated from the 2D
   image. This is extracting from the un-subtracted image (sky still
   present), which is normal for boxcar extraction.

3. **Equalize all extracted spectra**: Divide each fiber's 1D spectrum by
   `scale_factor_i * superflat(lambda) * fiberflat_i(lambda)`. Ivar scales as
   `correction^2`. After equalization, all fibers (sky and science) are on
   a common flux scale.

4. **Build B-spline sky model** from equalized sky fibers:
   - Combine all equalized sky fiber spectra into a super-sampled 1D array
     sorted by wavelength
   - Construct **smoothed** inverse-variance weight vector: boxcar smooth
     ~100 pixels in continuum regions, ~2 pixels near bright emission lines.
     This is critical per MaNGA experience -- using raw ivar causes Poisson
     scatter to modulate the weights and bias the fit toward lower values,
     resulting in systematic sky undersubtraction.
   - B-spline fit with iterative sigma-clipping rejection
   - Knot spacing is grating-dependent (e.g., 1.2 A for 270 gpm, 0.6 A for
     600 gpm, 0.3 A for 1000 gpm)

5. **Subtract sky from each fiber's 1D spectrum**: Evaluate the B-spline sky
   model on each fiber's native wavelength grid and subtract from the
   equalized spectrum. Both sky and science fibers are subtracted (sky fiber
   residuals serve as a quality check). Variance of the sky model from the
   B-spline fit covariance is added to the spectrum variance.

6. **Populate SpecObj outputs**: The sky-subtracted equalized 1D spectra
   become the `BOX_COUNTS` (sky-subtracted) and `BOX_COUNTS_SKY` (the sky
   model evaluated for that fiber) attributes. Ivar from the equalized
   extraction + sky model variance.

7. **Optimal extraction** (FiberExtract): For optimal extraction, the sky
   model is passed to `extract.extract_optimal()` which accounts for it in
   the noise weighting. The optimal extraction operates on the
   un-subtracted 2D image but with the per-fiber sky spectrum informing the
   variance model. Empirical spatial profiles from the flat field are used.
   Populates `OPT_*` SpecObj attributes.

8. **Diagnostic 2D sky image** (optional): A 2D sky image can be
   reconstructed by projecting each fiber's scaled sky spectrum back through
   its extraction aperture. This is for visualization and QA only, not used
   in the subtraction.

### Component 3: Integration

**Calibrations dispatch** (`calibrations.py`):
- Override `IFUCalibrations.get_flats()`: when `pypeline == 'Fiber'`,
  instantiate `FiberFlatField` instead of `FlatField`
- Fiber-specific flat products stored in the standard calibrations directory

**2D image processing** (`rawimage.py`):
- `apply_flat_fielding()` applies only `pixelflat_norm` for Fiber pypeline
- No illumination flat, no spectral illumination correction
- The 2D image retains lamp spectrum and fiber throughput differences

**FiberFindObjects.run()**: Implements steps 1-6 above, replacing the current
`global_skysub` / `joint_skysub` call. Extracts all fibers, equalizes,
builds sky model, subtracts in 1D. Returns SpecObjs with sky-subtracted
boxcar extractions.

**FiberExtract**: Performs optimal extraction on the un-subtracted 2D image,
using the per-fiber sky model from FiberFindObjects for noise weighting
(step 7). No post-extraction throughput correction needed.

**What stays unchanged**: Block-slit edge detection, per-block wavelength
calibration, fiber identification via cross-correlation, extraction machinery,
spec1d output format.

**What is removed**:
- `joint_skysub` / `global_skysub` for Fiber pypeline
- `illum_profile_spectral_poly` for Fiber pypeline
- `fiber_illumination.fits` dependency (replaced by superflat-derived
  throughput ratio)
- Post-extraction throughput correction in `apply_throughput_corrections()`
- `skyline_illum_correct()` during sky subtraction

## Variance Propagation

Variance is tracked through every step:

1. **Extraction**: `var_1d = moment1d(var_2d, trace, aperture)` -- standard
   boxcar variance propagation from 2D pixel variances.

2. **Equalization**: Dividing by correction `c` scales variance by `c^2`:
   `ivar_equalized = ivar_extracted * c^2`.

3. **Sky model**: B-spline fit variance from the fit covariance. Typically
   small relative to Poisson noise.

4. **Sky subtraction (1D)**: `var_subtracted = var_equalized + var_sky_model`.
   The sky model variance is propagated from the B-spline fit.

5. **Optimal extraction**: Operates on un-subtracted 2D image with sky model
   informing the noise weighting. Variance model uses actual per-pixel
   counts (sky + source) via `variance_model()`.

## Success Criteria

1. Pipeline completes without crashes (current `joint_skysub` crashes)
2. Sky subtraction residuals approach Poisson-noise limit, comparable to
   IDL pipeline
3. Noise model is self-consistent: measured noise matches predicted noise
   from variance arrays
4. All fibers extracted with both boxcar and optimal extraction
5. Runtime comparable to or better than current block-slit implementation
   (~51 minutes for JADES_1031022)
6. Throughput ratio from superflat consistent with known sky/science fiber
   aperture difference

## Open Questions

1. **Optimal extraction with un-subtracted image**: PypeIt's
   `extract.extract_optimal()` normally operates on a sky-subtracted image.
   For the Fiber pypeline, it will run on the un-subtracted image with the
   sky model passed separately. This may require adaptation of the
   extraction code to handle the sky model correctly in the noise weighting.

2. **Optimal extraction of sky fibers**: Currently only boxcar is used for
   sky model construction. Optimal extraction could improve S/N of the sky
   model but may introduce profile-dependent biases.

3. **Spatial sky variation**: The current design uses a single sky model for
   all fibers. If sky varies across the IFU field, per-block scale factors
   (like MaNGA's per-harness scaling) may be needed.
