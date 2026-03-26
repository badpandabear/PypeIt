# Hybrid Fiber-as-Object Extraction Design

**Date:** 2026-03-25
**Branch:** binospec_updates
**Status:** Draft

## Problem Statement

PypeIt's `Fiber` pypeline currently treats each fiber as an individual slit,
producing ~360 narrow slits (~3.4px wide) per detector for Binospec IFU.  This
causes several problems:

1. **Flux loss**: Fiber spacing is ~6.6px but detected slit width is ~3.4px,
   so nearly half the fiber flux falls in inter-slit gaps.
2. **Flat field failures**: Narrow slits cause the spatial normalization to
   fail for ~20 fibers per detector (`BADFLATCALIB`).
3. **Illumination correction inconsistency**: Sky-line-based throughput
   correction is computed within narrow slit boundaries, but extraction needs
   wider apertures.  Widening extraction without widening the correction
   creates inconsistencies.
4. **Performance**: Wavelength calibration runs identify/reidentify/fit-tilts
   720 times (once per fiber-slit across 2 detectors), taking hours.

## Solution: Hybrid Fiber-as-Object Model

Treat each physical fiber **block** as a slit and each **fiber** as an object
within that slit.  This maps directly to how PypeIt handles multi-slit
observations where multiple objects appear within a single slit.

For Binospec IFU, this produces 21 block-slits per detector (5 sky blocks of
8 fibers + 16 science blocks of 20 fibers) instead of 360 individual fiber
slits.  Inter-block gaps of ~66-74px provide clean slit boundaries.

The design is general for any fiber spectrograph:

- **With block structure** (e.g., Binospec IFU): block-level slit edges,
  fibers as objects within each block-slit.
- **Without block structure**: one slit spanning the full detector, all fibers
  as objects (long-slit analogy).

### Binospec IFU Block Structure (DET01)

| Block | Fibers | Span (px) | Gap to next (px) | Type    |
|-------|--------|-----------|-------------------|---------|
| 1     | 8      | 47        | 66                | Sky     |
| 2     | 20     | 126       | 70                | Science |
| 3     | 20     | 127       | 69                | Science |
| 4     | 20     | 126       | 70                | Science |
| 5     | 20     | 126       | 71                | Science |
| 6     | 8      | 46        | 70                | Sky     |
| 7     | 20     | 125       | 68                | Science |
| 8     | 20     | 127       | 68                | Science |
| 9     | 20     | 126       | 69                | Science |
| 10    | 20     | 126       | 69                | Science |
| 11    | 8      | 46        | 69                | Sky     |
| 12    | 20     | 126       | 70                | Science |
| 13    | 20     | 125       | 69                | Science |
| 14    | 20     | 126       | 70                | Science |
| 15    | 20     | 126       | 66                | Science |
| 16    | 8      | 47        | 74                | Sky     |
| 17    | 20     | 126       | 69                | Science |
| 18    | 20     | 126       | 70                | Science |
| 19    | 20     | 126       | 69                | Science |
| 20    | 20     | 126       | 66                | Science |
| 21    | 8      | 47        | --                | Sky     |

Sky blocks (1, 6, 11, 16, 21) contain 8 dedicated sky fibers each = 40 total.

## Processing Flow

```
1. Edge detection -> 21 block-slits per detector
   (existing Sobel filter finds block edges at ~70px gaps)

2. Flat field processing:
   a. Spectral normalization only (remove lamp blaze/continuum)
   b. No spatial illumination correction
   c. Output: pixelflat (pixel-to-pixel QE variations only)
   d. Build empirical fiber profiles from spectrally-normalized flat
   e. Measure per-fiber flat flux integrals for throughput calibration

3. Wavelength calibration:
   - Per-block-slit (21 per detector = 42 total, vs 720 currently)
   - identify -> reidentify -> fit tilts, same as MultiSlit per-slit
   - Tilts image interpolates across block, equivalent result

4. Science frame processing:
   a. Apply pixelflat (no spatial illumination)
   b. Scattered light correction (unchanged from current)
   c. Fiber identification via cross-correlation against reference profile
   d. Sky subtraction (see detailed flow below)
   e. Extract science fibers from sky-subtracted image using MultiSlit
      machinery with flat-derived profiles
   f. Apply throughput corrections to extracted 1D spectra
```

## Sky Subtraction

The sky subtraction flow changes significantly from the current pixel-level
approach.  Sky fibers (bare fibers) have substantially higher throughput than
science fibers (lenslet-fed), and this difference is not fully captured by
`fiber_illumination.fits`.  The inter-fiber gaps within blocks contain only
scattered light and cross-talk, not sky signal.

### Sky Subtraction Flow

```
1. Extract sky fibers from sky block-slits
   - Use flat-derived empirical profiles (same as science extraction)
   - Boxcar + optimal extraction of 40 sky fibers per side
   - Produces 40 extracted 1D sky spectra

2. Apply throughput corrections to sky fiber spectra
   a. Flat-based bulk scale factor (single scalar):
      ratio = avg(science fiber flat flux) / avg(sky fiber flat flux)
      This captures the bulk bare-vs-lenslet throughput difference.
   b. fiber_illumination.fits per-fiber corrections:
      accounts for fiber-to-fiber variations within the sky fibers.
   Both corrections put sky fiber spectra on the same throughput
   scale as science fibers.

3. Build sky model from corrected sky spectra
   - B-spline fit across 40 throughput-corrected sky fiber spectra
   - Grating-dependent knot spacing (1.05/0.50/0.35 Ang for 270/600/1000)
   - Joint fit in wavelength, same algorithm as current joint_skysub

4. Project sky model back into 2D
   - Evaluate B-spline sky model at each pixel's wavelength
   - Scale by each science fiber's throughput (from flat + fiber_illumination)
   - Produces 2D sky model image for subtraction and diagnostics (spec2d)

5. Subtract 2D sky model from science image
   - Update variance model with sky contribution
   - Sky-subtracted image used for science fiber extraction

6. Extract science fibers from sky-subtracted image
   - MultiSlit machinery with flat-derived profiles
   - Boxcar + Horne (1986) optimal extraction per fiber
```

### Why Extract Sky Fibers First

- Sky fibers must be extracted and throughput-corrected before building the
  sky model, because the bare-vs-lenslet throughput difference is large
- Working with extracted 1D spectra makes the throughput corrections simpler
  and more robust than 2D pixel-level corrections
- The 2D sky model projected back into the image preserves proper variance
  weighting during science fiber optimal extraction
- The 2D sky model in spec2d output provides useful diagnostics

### Throughput Correction Chain

Two levels of throughput correction, applied in order:

1. **Flat-based bulk scale** (single scalar per side):
   `avg(science fiber flat flux) / avg(sky fiber flat flux)`
   Captures the dominant bare-vs-lenslet throughput difference.

2. **`fiber_illumination.fits`** (per-fiber scalar):
   Captures fiber-to-fiber variations within each group (sky and science).

Both are applied to sky fiber spectra before building the sky model, putting
them on the science fiber throughput scale.  The same `fiber_illumination.fits`
corrections are applied to science fiber spectra post-extraction.

### Comparison with Current Approach

| Aspect              | Current                          | New                               |
|---------------------|----------------------------------|-----------------------------------|
| Sky input           | 2D pixels within sky fiber slits | Extracted 1D sky fiber spectra    |
| Throughput handling  | `skyline_illum_correct` on 2D image | Flat scale + fiber_illum on 1D spectra |
| Sky model           | B-spline fit on 2D pixels        | B-spline fit on corrected 1D spectra |
| Subtraction         | 2D model evaluated at all pixels | 2D model projected from 1D model  |
| Diagnostics         | 2D sky model in spec2d           | Same (preserved)                  |

## Detailed Design by Module

### Edge Detection and Slit Definition

**Current behavior**: Sobel filter detects 720+ edges (left/right for each
fiber), producing ~360 narrow slits per detector.

**New behavior**: Edge detection parameters tuned so the Sobel filter finds
block boundaries at the ~70px inter-block gaps.  This produces 21 block-slits
per detector.  The spectrograph class provides detection parameters
(e.g., minimum slit width, edge separation threshold) to ensure block-level
detection rather than fiber-level.

**Affected files**: `pypeit/edgetrace.py` (detection parameters),
`pypeit/spectrographs/mmt_binospec.py` (spectrograph-specific edge parameters).

### Flat Field

**Current behavior**: Three-stage normalization (spectral + spatial
illumination + 2D residual).  `modify_pixelflat()` bakes
`fiber_illumination.fits` into the pixelflat.  `compute_skyline_illum()`
produces a 2D correction image applied to the science frame before extraction.

**New behavior**:
- Spectral normalization only: remove lamp blaze/continuum via B-spline fit
  along the spectral axis.  Skip spatial illumination entirely
  (`use_illumflat = False`).
- The pixelflat captures pixel-to-pixel QE variations only.
- `modify_pixelflat()` no longer bakes `fiber_illumination.fits` into the flat.
- Empirical fiber profiles extracted from the spectrally-normalized flat:
  for each fiber within each block-slit, median-collapse the fiber's
  cross-section and normalize to unit sum.
- Measure per-fiber integrated flat flux for throughput calibration
  (used to compute the bulk sky/science throughput ratio and per-fiber
  relative throughputs).

**Affected files**: `pypeit/flatfield.py` (skip spatial illumination for
Fiber pypeline), `pypeit/spectrographs/mmt_binospec.py` (modify_pixelflat,
new flat flux measurement, profile building).

### Wavelength Calibration

**Current behavior**: Runs identify/reidentify/fit-tilts once per fiber-slit
(720 times across 2 detectors).

**New behavior**: Runs per block-slit (42 times across 2 detectors).  Each
block-slit contains multiple fibers, so the arc line identification and tilt
fitting operates across the full spatial extent of the block.  The resulting
tilts image should be equivalent since it interpolates across the slit.

**Performance impact**: ~17x reduction in wavelength calibration calls.

**Affected files**: No code changes needed — the existing per-slit wavecalib
machinery operates on whatever slits are defined.

### Object Finding (`FiberFindObjects`)

**Current behavior**: Creates one SpecObj per slit (one fiber = one slit =
one object).  Each SpecObj has `TRACE_SPAT` at the slit center, `FWHM` from
slit width, `BOX_R_PIX` from slit half-width.

**New behavior**: Two-phase object finding.

**Phase 1 — Sky fiber extraction** (before science object finding):
1. Identify sky block-slits (blocks 1, 6, 11, 16, 21).
2. Create SpecObjs for sky fibers within each sky block-slit.
3. Extract sky fibers using flat-derived profiles.
4. Apply throughput corrections (flat bulk scale + `fiber_illumination.fits`).
5. Build B-spline sky model from corrected sky spectra.
6. Project sky model back into 2D and subtract from science image.

**Phase 2 — Science fiber object finding** (on sky-subtracted image):
1. For each science block-slit, identify fibers from reference profile.
2. Create one SpecObj per fiber with:
   - `TRACE_SPAT`: fiber center position within the block-slit
   - `FWHM`: fiber profile FWHM (~6.6px, full fiber spacing)
   - `BOX_R_PIX`: half the inter-fiber spacing (adjacent apertures touch)
   - `maskwidth`: set so adjacent fiber masks touch
   - Fiber metadata: `MASKDEF_ID`, `MASKDEF_OBJNAME`, fiber type

**Affected files**: `pypeit/find_objects.py` (`FiberFindObjects` class).

### Extraction (`FiberExtract`)

**Current behavior**: Independent per-fiber extraction.  Boxcar uses
`moment1d` within narrow slit mask.  Optimal extraction uses empirical
profiles from flat, applied within narrow slit pixel selection.

**New behavior**: Delegates to MultiSlit extraction machinery within each
block-slit:
1. For each science block-slit, gather all fiber SpecObjs.
2. Pass flat-derived spatial profiles for each fiber to the extraction
   routine.
3. MultiSlit's multi-object extraction handles simultaneous profile fitting,
   naturally accounting for cross-talk between adjacent fibers.
4. **No local sky subtraction** — the global sky model is already subtracted.
   Configure the extraction to use the sky-subtracted image directly.
5. Boxcar extraction uses `BOX_R_PIX` = half inter-fiber spacing.
6. Optimal extraction uses flat-derived profiles with Horne (1986) weighting.

**Fallback**: If the MultiSlit machinery proves too constraining (e.g.,
assumptions about local sky fitting that can't be cleanly disabled), fall
back to reimplementing the multi-object extraction directly in
`FiberExtract`.

**Affected files**: `pypeit/extraction.py` (`FiberExtract` class),
potentially `pypeit/core/skysub.py` (if modifications needed to skip local
sky).

### Throughput Corrections (Post-Extraction)

**Current behavior**: `fiber_illumination.fits` baked into pixelflat via
`modify_pixelflat()`.  `compute_skyline_illum()` produces 2D correction image
applied before extraction.

**New behavior**: Corrections applied at two stages:

**Before sky model** (to sky fiber spectra):
1. Flat-based bulk scale factor (single scalar per side).
2. `fiber_illumination.fits` per-fiber corrections.

**After science extraction** (to science fiber spectra):
1. `fiber_illumination.fits` per-fiber corrections.
2. Sky-line-based correction (per-exposure, computed from extracted spectra).
3. Divide each fiber's extracted counts by its combined correction.
4. Propagate through inverse variance.

This approach is cleaner because:
- Corrections are applied uniformly to all extracted flux, not limited to
  pixels within narrow slit boundaries.
- The correction is a simple scalar per fiber, avoiding 2D image artifacts.
- Sky-line correction can be computed from extracted spectra rather than
  the 2D image, which is simpler and more robust.

**Affected files**: `pypeit/spectrographs/mmt_binospec.py` (new method for
post-extraction correction), `pypeit/extraction.py` or `pypeit/reduce.py`
(call the correction after extraction).

### Spectrograph Configuration

The spectrograph class provides fiber-specific configuration:

- **Block assignments**: Which fibers belong to which blocks, from the
  reference profile's `FIB_BLOCK` column.
- **Sky fiber identification**: By `FIB_NAME` prefix (`SKY*`).
- **Edge detection parameters**: Tuned for block-level detection
  (e.g., `edge_thresh`, `minimum_slit_length`, `length_range`).
- **Flat parameters**: `use_illumflat = False`.
- **Throughput corrections**: `fiber_illumination.fits` loading, flat-based
  bulk scale computation, `compute_skyline_illum()` (adapted for 1D spectra).
- **Fiber metadata**: `get_fiber_metadata()` adapted to map objects within
  block-slits to physical fiber IDs.

**Affected files**: `pypeit/spectrographs/mmt_binospec.py`,
`pypeit/par/pypeitpar.py` (if new parameters needed).

## What Does NOT Change

- Scattered light correction (already per-frame for Binospec IFU)
- Spectral flexure correction (already disabled for Binospec)
- Cube building (`binospec_ifu_cube.py`) — still consumes spec1d/spec2d
- Core MultiSlit extraction algorithms (`pypeit/core/skysub.py`,
  `pypeit/core/extract.py`)
- `SlicerIFU` pypeline (unaffected)
- `BADFLATCALIB` issue: eliminated because block-slits are ~126px wide,
  so flat field spatial normalization is not applied.  No need for the
  `BADFLATCALIB -> SKIPFLATCALIB` downgrade that was in the stash.

## Performance Impact

| Step                  | Current (per detector) | New (per detector) | Speedup |
|-----------------------|------------------------|--------------------|---------|
| Edge detection        | ~360 slits             | ~21 slits          | ~17x    |
| Wavelength calibration| 360 identify+reidentify| 21 identify+reidentify | ~17x |
| Flat field            | 360 spatial fits       | 0 (no spatial)     | N/A     |
| Sky subtraction       | B-spline on 2D pixels  | Extract + B-spline on 1D | ~similar |
| Extraction            | 360 independent        | 21 multi-object    | ~similar|

The wavelength calibration speedup alone should cut the ~5.5 hour runtime
substantially, as it is the dominant cost in the current pipeline.

## Risks and Mitigations

1. **MultiSlit extraction assumes local sky**: The `local_skysub_extract()`
   function in `core/skysub.py` jointly fits sky and object profiles.  We
   need to either (a) pass the already-subtracted image and configure it to
   skip the sky fitting step, or (b) bypass `local_skysub_extract()` and
   call the extraction functions directly.
   *Mitigation*: Investigate the `no_local_sky` or similar options in the
   MultiSlit machinery.  Fallback to direct implementation in FiberExtract
   if needed.

2. **Edge detection parameter tuning**: Current Sobel filter parameters are
   tuned for fiber-level edges.  Block-level edges are much more prominent
   (70px gaps vs ~3px fiber spacing), so detection should be easier, but
   parameters need adjustment.
   *Mitigation*: Test with existing calibration data; the block gaps are
   large enough that a wide range of threshold values should work.

3. **Cross-talk modeling**: Simultaneous multi-object fitting naturally
   handles cross-talk between adjacent fibers, but the accuracy depends on
   the profile model quality.  If flat-derived profiles don't capture the
   true PSF well enough, cross-talk could contaminate neighboring fibers.
   *Mitigation*: Compare extracted spectra with and without simultaneous
   fitting; fall back to independent extraction if cross-talk modeling
   degrades results.

4. **Fiber matching within blocks**: Currently fiber identification matches
   individual fiber traces to the reference.  With block-slits, we need to
   identify individual fibers *within* each block from their positions
   relative to the block edges.
   *Mitigation*: The reference profile's `FIB_BLOCK` column directly maps
   fibers to blocks.  Within each block, fibers are well-ordered by pixel
   position, so matching is straightforward.

5. **Sky model projection to 2D**: Projecting the 1D B-spline sky model
   back into 2D requires knowing each pixel's wavelength (from tilts) and
   which fiber it belongs to (for throughput scaling).  This is more complex
   than the current approach where the sky model is fit and evaluated in 2D
   directly.
   *Mitigation*: The tilts image and fiber profile assignments from the flat
   provide all necessary information.  The projection is a straightforward
   evaluation of the B-spline at each pixel's wavelength, multiplied by the
   fiber's throughput.

## Success Criteria

1. All fibers extracted (no `BADFLATCALIB` exclusions)
2. Extracted flux captures full fiber profile (no wing loss)
3. Fiber-to-fiber throughput variation in extracted spectra matches raw data
   (before throughput corrections)
4. Sky subtraction quality equal to or better than current implementation
5. Pipeline runtime reduced by at least 5x for wavelength calibration
6. Cube quality comparable to or better than current output
7. 2D sky model in spec2d output provides clean diagnostic images
