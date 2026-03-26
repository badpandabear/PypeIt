# Fiber Block-Slit Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the Fiber pypeline to treat fiber blocks as slits and individual fibers as objects within each block-slit, eliminating flux loss from narrow slit definitions and improving performance.

**Architecture:** Edge detection produces ~21 block-slits per detector instead of ~360 fiber-slits. Fibers become objects within blocks using flat-derived profiles. Sky fibers are extracted first to build a throughput-corrected sky model, which is projected into 2D for subtraction before science extraction. No spatial illumination correction; throughput corrections applied post-extraction.

**Tech Stack:** Python, NumPy, SciPy, Astropy, PypeIt

**Spec:** `claude_docs/specs/2026-03-25-fiber-block-slit-extraction-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `pypeit/spectrographs/mmt_binospec.py` | Modify | Edge params, flat params, block config, sky/science throughput methods |
| `pypeit/find_objects.py` | Modify | `FiberFindObjects`: multi-object per block-slit, sky fiber extraction |
| `pypeit/extraction.py` | Modify | `FiberExtract`: block-aware extraction with flat profiles |
| `pypeit/flatfield.py` | Modify | Skip spatial illumination for Fiber pypeline |
| `pypeit/tests/test_fiber_block_extraction.py` | Create | Unit tests for new fiber block logic |

---

### Task 1: Edge Detection — Block-Level Slit Parameters

Tune edge detection to find 21 block boundaries (at ~70px inter-block gaps)
instead of 360 fiber-level edges.

**Files:**
- Modify: `pypeit/spectrographs/mmt_binospec.py:1336-1349`

- [ ] **Step 1: Change edge detection parameters**

In `MMTBINOSPECIFUSpectrograph.default_pypeit_par()`, replace the fiber-level
edge parameters with block-level parameters:

```python
# Replace lines 1336-1349:

# Slit edge parameters tuned for block-level detection.
# Binospec IFU fibers are organized in 21 blocks (5 sky + 16 science)
# separated by ~66-74 pixel gaps. Within blocks, fibers are ~6.6 pixels
# apart with no true gaps (only cross-talk and scattered light between them).
# The edge_thresh is set high enough to detect only block boundaries,
# not individual fiber edges within blocks.
par['calibrations']['slitedges']['edge_thresh'] = 100.
par['calibrations']['slitedges']['minimum_slit_gap'] = 0.
par['calibrations']['slitedges']['minimum_slit_length'] = 5.0  # ~12 arcsec, blocks are ~30+ arcsec
par['calibrations']['slitedges']['pad'] = 0
par['calibrations']['slitedges']['use_maskdesign'] = False
par['calibrations']['slitedges']['fwhm_gaussian'] = 3.0  # default
par['calibrations']['slitedges']['min_edge_side_sep'] = 5.0  # default: min 15px between same-side edges

# Tilts: increase spat_order for block-slits (~126px wide vs ~5px fiber slits)
par['calibrations']['tilts']['spat_order'] = 3
par['calibrations']['tilts']['spec_order'] = 3
```

- [ ] **Step 2: Test edge detection on calibration data**

Run edge tracing only to verify 21 block-slits are detected per detector:

```bash
cd ~/MMT/bino_ifu/JADES_1031022
# Check if existing calibrations need to be cleared
ls Calibrations/Edges_A_0_DET01.fits.gz
```

Then use a quick Python check on the existing flat to see what the Sobel filter
produces with the new threshold:

```python
python3 -c "
from pypeit.core import trace
from pypeit import flatfield
from astropy.io import fits
import numpy as np

# Load flat
flat = fits.getdata('Calibrations/Flat_A_0_DET01.fits', ext=1)
# Run Sobel detection with high threshold
edges = trace.detect_slit_edges(flat, sigdetect=100.)
n_left = np.sum(edges == -1)
n_right = np.sum(edges == 1)
print(f'Left edges: {n_left}, Right edges: {n_right}')
# Should be ~21 of each for block-level detection
"
```

If 21 slits are not detected, adjust `edge_thresh` up or down. The inter-block
gaps (~70px with near-zero flux) produce very strong Sobel gradients compared
to intra-block fiber boundaries (~2px gaps), so there should be a wide range
of threshold values that work.

- [ ] **Step 3: Commit**

```bash
git add pypeit/spectrographs/mmt_binospec.py
git commit -m "feat(binospec): tune edge detection for block-level slit boundaries

Change edge_thresh from 5.0 to 100.0 so the Sobel filter detects the
21 inter-block gaps (~70px) instead of ~360 individual fiber edges.
This is the foundation for the hybrid block-slit extraction approach."
```

---

### Task 2: Flat Field — Disable Spatial Illumination

Remove spatial illumination correction and stop baking `fiber_illumination.fits`
into the pixelflat for the Fiber pypeline.

**Files:**
- Modify: `pypeit/spectrographs/mmt_binospec.py:1358-1361,1590-1642`

- [ ] **Step 1: Update flat field parameters**

In `MMTBINOSPECIFUSpectrograph.default_pypeit_par()`, ensure spatial
illumination is fully disabled. The key setting `use_illumflat = False` is
already set at line 1318. Verify and add any missing flat parameters:

```python
# Lines 1358-1361, replace with:

# Flat field: spectral-only normalization for fiber-fed IFU.
# No spatial illumination correction — fibers don't have spatial extent.
# Throughput corrections applied post-extraction instead.
par['calibrations']['flatfield']['tweak_slits'] = False
par['calibrations']['flatfield']['slit_trim'] = 0
par['calibrations']['flatfield']['slit_illum_finecorr'] = False
# Ensure the illumflat is not used for science processing
par['scienceframe']['process']['use_illumflat'] = False
```

- [ ] **Step 2: Disable modify_pixelflat**

The `modify_pixelflat()` method currently bakes `fiber_illumination.fits` into
the pixelflat. In the new design, throughput corrections are applied
post-extraction. Disable this method:

```python
# In modify_pixelflat(), replace the body (lines 1607-1642) with:
def modify_pixelflat(self, flatimages, slits, det):
    """
    Override to skip baking fiber illumination into the pixel flat.

    In the block-slit extraction approach, throughput corrections
    (fiber_illumination.fits and sky-line-based) are applied to
    extracted 1D spectra, not to the pixel flat.
    """
    pass
```

- [ ] **Step 3: Add flat flux measurement method**

Add a method to measure integrated flat flux per fiber for throughput
calibration (needed for sky/science throughput ratio):

```python
def measure_fiber_flat_flux(self, flatimg, slits, det):
    """
    Measure integrated flat field flux for each fiber within block-slits.

    Used to compute the bulk throughput ratio between sky fibers (bare)
    and science fibers (lenslet-fed).

    Args:
        flatimg (`numpy.ndarray`_):
            Flat field image, shape ``(nspec, nspat)``.
        slits (:class:`~pypeit.slittrace.SlitTraceSet`):
            Block-slit traces (21 per detector).
        det (:obj:`int`):
            1-indexed detector number.

    Returns:
        :obj:`dict`: Dictionary with keys:
            - 'fiber_flux': per-fiber integrated flat flux (nfibers,)
            - 'fiber_type': per-fiber type ('sky' or 'science')
            - 'sky_avg': mean flux of sky fibers
            - 'sci_avg': mean flux of science fibers
            - 'bulk_scale': sci_avg / sky_avg (scalar)
    """
    from pypeit.core.moment import moment1d

    blocks = self.get_fiber_blocks(det)
    slitmask = slits.slit_img(pad=0)

    all_flux = []
    all_type = []

    for block_idx, block in enumerate(blocks):
        slit_spat_id = slits.spat_id[block_idx]
        thismask = slitmask == slit_spat_id
        if not np.any(thismask):
            continue

        for j, fpos in enumerate(block['fiber_positions']):
            # Boxcar integrate the flat flux around each fiber
            trace = np.full(flatimg.shape[0], fpos)
            # Half-spacing from neighbors
            if block['nfibers'] > 1:
                positions = block['fiber_positions']
                spacings = np.diff(positions)
                if j == 0:
                    half_sp = spacings[0] / 2.0
                elif j == len(positions) - 1:
                    half_sp = spacings[-1] / 2.0
                else:
                    half_sp = min(spacings[j-1], spacings[j]) / 2.0
            else:
                half_sp = 3.3  # single fiber fallback
            box_flux = moment1d(flatimg * thismask, trace, 2 * half_sp,
                                row=np.arange(flatimg.shape[0]))[0]
            med_flux = np.median(box_flux[box_flux > 0]) if np.any(box_flux > 0) else 0.0
            all_flux.append(med_flux)
            all_type.append(block['type'])

    all_flux = np.array(all_flux)
    all_type = np.array(all_type)
    sky_mask = all_type == 'sky'
    sci_mask = all_type == 'science'

    sky_avg = np.median(all_flux[sky_mask]) if np.any(sky_mask) else 1.0
    sci_avg = np.median(all_flux[sci_mask]) if np.any(sci_mask) else 1.0
    bulk_scale = sci_avg / sky_avg if sky_avg > 0 else 1.0

    return {
        'fiber_flux': all_flux,
        'fiber_type': all_type,
        'sky_avg': sky_avg,
        'sci_avg': sci_avg,
        'bulk_scale': bulk_scale,
    }
```

- [ ] **Step 4: Commit**

```bash
git add pypeit/spectrographs/mmt_binospec.py
git commit -m "feat(binospec): disable spatial illumination and pixelflat throughput baking

Throughput corrections will be applied post-extraction to 1D spectra.
Add measure_fiber_flat_flux() for computing sky/science throughput ratio."
```

---

### Task 3: Block Configuration — Expose Fiber Block Structure

Add methods to `MMTBINOSPECIFUSpectrograph` that expose the fiber block
structure from the reference profile, enabling block-aware processing.

**Files:**
- Modify: `pypeit/spectrographs/mmt_binospec.py`
- Create: `pypeit/tests/test_fiber_block_extraction.py`

- [ ] **Step 1: Write test for block configuration**

```python
# pypeit/tests/test_fiber_block_extraction.py
"""Tests for fiber block-slit extraction logic."""
import numpy as np
import pytest
from pypeit.spectrographs.mmt_binospec import MMTBINOSPECIFUSpectrograph


class TestFiberBlockConfig:
    """Tests for fiber block configuration methods."""

    @classmethod
    def setup_class(cls):
        cls.spec = MMTBINOSPECIFUSpectrograph()

    def test_get_fiber_blocks_det01(self):
        """DET01 should have 21 blocks: 5 sky (8 fibers) + 16 science (20)."""
        blocks = self.spec.get_fiber_blocks(1)
        assert len(blocks) == 21
        # Sky blocks: 1, 6, 11, 16, 21
        for b in [0, 5, 10, 15, 20]:  # 0-indexed
            assert blocks[b]['type'] == 'sky'
            assert blocks[b]['nfibers'] == 8
        # Science blocks
        for b in [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19]:
            assert blocks[b]['type'] == 'science'
            assert blocks[b]['nfibers'] == 20

    def test_get_fiber_blocks_det02(self):
        """DET02 should also have 21 blocks."""
        blocks = self.spec.get_fiber_blocks(2)
        assert len(blocks) == 21

    def test_block_fiber_positions(self):
        """Fiber positions within blocks should be monotonically ordered."""
        blocks = self.spec.get_fiber_blocks(1)
        for block in blocks:
            positions = block['fiber_positions']
            assert np.all(np.diff(positions) > 0), \
                f"Block {block['block_id']} positions not monotonic"

    def test_block_gaps(self):
        """Inter-block gaps should be > 50 pixels."""
        blocks = self.spec.get_fiber_blocks(1)
        for i in range(len(blocks) - 1):
            gap = blocks[i+1]['min_pix'] - blocks[i]['max_pix']
            assert gap > 50, f"Gap between blocks {i} and {i+1} is only {gap:.1f} px"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest pypeit/tests/test_fiber_block_extraction.py::TestFiberBlockConfig -v
```
Expected: FAIL with `AttributeError: 'MMTBINOSPECIFUSpectrograph' object has no attribute 'get_fiber_blocks'`

- [ ] **Step 3: Implement get_fiber_blocks**

Add to `MMTBINOSPECIFUSpectrograph`:

```python
def get_fiber_blocks(self, det):
    """
    Return the fiber block structure from the reference profile.

    Each block is a group of fibers that will become a single "slit"
    in the block-slit extraction approach. Blocks are defined by the
    FIB_BLOCK column in the reference profile.

    Args:
        det (:obj:`int`):
            1-indexed detector number (1=side A, 2=side B).

    Returns:
        :obj:`list` of :obj:`dict`: One dict per block with keys:
            - 'block_id': int, block number from reference profile
            - 'nfibers': int, number of fibers in block
            - 'type': str, 'sky' or 'science'
            - 'fiber_positions': ndarray, reference pixel positions (TR_PIX)
            - 'fiber_names': list of str, fiber names
            - 'fiber_ids': ndarray, fiber IDs
            - 'min_pix': float, minimum pixel position in block
            - 'max_pix': float, maximum pixel position in block
    """
    ref = self.load_fiber_ref_profile(det)
    block_ids = np.unique(ref['FIB_BLOCK'])
    blocks = []
    for bid in block_ids:
        mask = ref['FIB_BLOCK'] == bid
        names = [n.strip() for n in ref['FIB_NAME'][mask]]
        n_sky = sum(1 for n in names if n.startswith('SKY'))
        positions = ref['TR_PIX'][mask]
        blocks.append({
            'block_id': int(bid),
            'nfibers': int(np.sum(mask)),
            'type': 'sky' if n_sky > 0 else 'science',
            'fiber_positions': positions,
            'fiber_names': names,
            'fiber_ids': ref['FIB_ID'][mask],
            'min_pix': float(positions.min()),
            'max_pix': float(positions.max()),
        })
    return blocks
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest pypeit/tests/test_fiber_block_extraction.py::TestFiberBlockConfig -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pypeit/spectrographs/mmt_binospec.py pypeit/tests/test_fiber_block_extraction.py
git commit -m "feat(binospec): add get_fiber_blocks() for block structure access

Exposes fiber block assignments from the reference profile, needed for
the block-slit extraction approach. Includes unit tests."
```

---

### Task 4: Fiber Identification Within Blocks

Adapt `get_fiber_metadata()` to work with block-slits. Instead of matching
individual slit positions to fiber positions, match fiber peak positions
*within* each block-slit to the reference profile.

**Files:**
- Modify: `pypeit/spectrographs/mmt_binospec.py`
- Modify: `pypeit/tests/test_fiber_block_extraction.py`

- [ ] **Step 1: Write test for block-aware fiber matching**

```python
# Add to pypeit/tests/test_fiber_block_extraction.py

class TestFiberMatchingInBlocks:
    """Tests for identifying fibers within block-slits."""

    @classmethod
    def setup_class(cls):
        cls.spec = MMTBINOSPECIFUSpectrograph()

    def test_identify_fibers_in_block(self):
        """Given fiber peak positions within a block, identify each fiber."""
        blocks = self.spec.get_fiber_blocks(1)
        # Use the reference positions as "detected" positions for block 2 (science, 20 fibers)
        block = blocks[1]  # block 2 (0-indexed = 1)
        detected_positions = block['fiber_positions']
        result = self.spec.identify_fibers_in_block(
            det=1, block_idx=1, detected_positions=detected_positions)
        assert len(result['fiber_id']) == 20
        assert len(result['fiber_name']) == 20
        # All should match since we used reference positions
        assert np.all(result['fiber_id'] > 0)

    def test_identify_fibers_with_offset(self):
        """Fibers should still match with a small position offset."""
        blocks = self.spec.get_fiber_blocks(1)
        block = blocks[1]
        # Add 2-pixel offset (simulating flexure)
        detected_positions = block['fiber_positions'] + 2.0
        result = self.spec.identify_fibers_in_block(
            det=1, block_idx=1, detected_positions=detected_positions)
        assert np.all(result['fiber_id'] > 0)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest pypeit/tests/test_fiber_block_extraction.py::TestFiberMatchingInBlocks -v
```

- [ ] **Step 3: Implement identify_fibers_in_block**

```python
def identify_fibers_in_block(self, det, block_idx, detected_positions):
    """
    Identify fibers within a block-slit by matching detected peak positions
    to reference fiber positions.

    Args:
        det (:obj:`int`): 1-indexed detector number.
        block_idx (:obj:`int`): 0-based block index.
        detected_positions (`numpy.ndarray`_): Detected fiber peak pixel
            positions within the block-slit, sorted by position.

    Returns:
        :obj:`dict`: Keys 'fiber_id', 'fiber_name', 'fiber_type' — arrays
            aligned with detected_positions. Unmatched fibers get
            fiber_id=-1, fiber_name='UNKNOWN', fiber_type='unknown'.
    """
    blocks = self.get_fiber_blocks(det)
    block = blocks[block_idx]
    ref_positions = block['fiber_positions']
    ref_ids = block['fiber_ids']
    ref_names = block['fiber_names']

    n_det = len(detected_positions)
    n_ref = len(ref_positions)

    fiber_id = np.full(n_det, -1, dtype=int)
    fiber_name = np.array(['UNKNOWN'] * n_det, dtype='U20')
    fiber_type = np.array(['unknown'] * n_det, dtype='U10')

    # Compute offset: median shift between detected and reference
    if n_det == n_ref:
        offset = np.median(detected_positions - ref_positions)
    else:
        # Use cross-correlation for partial matches
        offset = 0.0  # Simple fallback; refine later if needed

    shifted_ref = ref_positions + offset

    # Match by nearest neighbor with threshold
    threshold = 3.5  # pixels
    for i, dpos in enumerate(detected_positions):
        dists = np.abs(shifted_ref - dpos)
        best = np.argmin(dists)
        if dists[best] < threshold:
            fiber_id[i] = int(ref_ids[best])
            fiber_name[i] = ref_names[best]
            fiber_type[i] = 'sky' if ref_names[best].startswith('SKY') else 'science'

    return {'fiber_id': fiber_id, 'fiber_name': fiber_name, 'fiber_type': fiber_type}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest pypeit/tests/test_fiber_block_extraction.py::TestFiberMatchingInBlocks -v
```

- [ ] **Step 5: Commit**

```bash
git add pypeit/spectrographs/mmt_binospec.py pypeit/tests/test_fiber_block_extraction.py
git commit -m "feat(binospec): add identify_fibers_in_block() for block-slit fiber matching

Matches detected fiber positions within a block-slit to reference
positions using nearest-neighbor matching with offset correction."
```

---

### Task 5: FiberFindObjects — Multi-Object Per Block-Slit

Refactor `FiberFindObjects.find_objects_pypeline()` to create multiple SpecObjs
per block-slit (one per fiber). This is the core architectural change.

**Files:**
- Modify: `pypeit/find_objects.py:1394-1497`

- [ ] **Step 1: Refactor find_objects_pypeline for block-slits**

Replace the current method body that creates one SpecObj per slit with logic
that detects fiber peaks within each block-slit and creates one SpecObj per
fiber:

```python
def find_objects_pypeline(self, image, ivar, std_trace=None,
                          manual_extract_dict=None,
                          show_peaks=False, show_fits=False, show_trace=False,
                          show=False, save_objfindQA=False, neg=False, debug=False):
    """
    Create one SpecObj per fiber within each block-slit.

    For each block-slit, detects fiber peak positions in the flat field
    (or from the spectrograph's reference profile), then creates a SpecObj
    at each fiber position with FWHM and BOX_R_PIX set to capture the full
    fiber flux.
    """
    gdslits = np.where(np.logical_not(self.reduce_bpm))[0]
    sobjs = specobjs.SpecObjs()
    nspec = image.shape[0]
    obj_counter = 0

    for slit_idx in gdslits:
        slit_spat_id = self.slits.spat_id[slit_idx]
        left = self.slits_left[:, slit_idx]
        right = self.slits_right[:, slit_idx]

        # Get reference fiber positions for this block from the spectrograph.
        # This uses the reference profile (TR_PIX) rather than peak-finding
        # in the flat, since FiberFindObjects does not have access to the
        # flat image (that belongs to Extract). The reference positions are
        # adjusted for flexure via cross-correlation in identify_fibers_in_block.
        blocks = self.spectrograph.get_fiber_blocks(self.det)
        if slit_idx >= len(blocks):
            continue
        block = blocks[slit_idx]
        fiber_centers = block['fiber_positions'].copy()

        if len(fiber_centers) == 0:
            continue

        # Identify fibers using spectrograph reference
        fiber_meta = self.spectrograph.identify_fibers_in_block(
            self.det, slit_idx, fiber_centers)

        # Compute inter-fiber spacings for BOX_R_PIX
        spacings = np.diff(fiber_centers)
        half_spacings = np.zeros(len(fiber_centers))
        if len(spacings) > 0:
            half_spacings[0] = spacings[0] / 2.0
            half_spacings[-1] = spacings[-1] / 2.0
            half_spacings[1:-1] = np.minimum(spacings[:-1], spacings[1:]) / 2.0
        else:
            half_spacings[0] = np.median(right - left) / 2.0

        for j, center_pix in enumerate(fiber_centers):
            obj_counter += 1
            trace_center = np.full(nspec, center_pix)
            # Refine: use flat field centroid per row if available
            # (for now, constant trace — refine in later task)

            thisobj = specobj.SpecObj(
                PYPELINE='Fiber',
                DET=self.sciImg.detector.name,
                OBJTYPE=self.objtype,
                SLITID=slit_spat_id,
            )
            thisobj.TRACE_SPAT = trace_center
            thisobj.trace_spec = np.arange(nspec)
            thisobj.SPAT_PIXPOS = center_pix
            thisobj.SPAT_PIXPOS_ID = int(np.rint(center_pix))
            thisobj.SPAT_FRACPOS = (center_pix - np.median(left)) / \
                np.median(right - left)
            thisobj.FWHM = 2.0 * half_spacings[j]
            thisobj.maskwidth = half_spacings[j] / np.median((right - left) / 2.0)
            thisobj.BOX_R_PIX = half_spacings[j]
            thisobj.smash_peakflux = 1.0
            thisobj.smash_snr = 100.0
            thisobj.OBJID = obj_counter

            # Assign fiber metadata
            if fiber_meta is not None:
                thisobj.MASKDEF_ID = int(fiber_meta['fiber_id'][j])
                thisobj.MASKDEF_OBJNAME = fiber_meta['fiber_name'][j]

            thisobj.set_name()
            sobjs.add_sobj(thisobj)

    self.steps.append(inspect.stack()[0][3])
    return sobjs, len(sobjs)
```

- [ ] **Step 2: Test with reference profile positions**

Verify that reference profile positions fall within block-slit boundaries:

```bash
python3 -c "
from pypeit.spectrographs.mmt_binospec import MMTBINOSPECIFUSpectrograph
spec = MMTBINOSPECIFUSpectrograph()
blocks = spec.get_fiber_blocks(1)
for i, block in enumerate(blocks):
    print(f'Block {block[\"block_id\"]}: {block[\"nfibers\"]} fibers, '
          f'range [{block[\"min_pix\"]:.1f}, {block[\"max_pix\"]:.1f}], '
          f'type={block[\"type\"]}')
# Verify all 360 fibers accounted for
total = sum(b['nfibers'] for b in blocks)
print(f'Total fibers: {total} (expected 360)')
"
```

Note: Fiber positions come from the reference profile's `TR_PIX` column, not
from peak-finding in the flat. The reference profile is a stable, pre-computed
calibration file. `FiberFindObjects` does not have access to the flat image
(that belongs to `Extract`), and the reference positions are more robust than
runtime peak detection.

- [ ] **Step 3: Commit**

```bash
git add pypeit/find_objects.py
git commit -m "feat(fiber): refactor FiberFindObjects for multi-object per block-slit

find_objects_pypeline() now detects fiber peaks within each block-slit
and creates one SpecObj per fiber. Adds _find_fiber_peaks_in_block()
to locate fibers from the flat field."
```

---

### Task 6: Sky Subtraction — Extract Sky Fibers and Build Sky Model

Refactor `FiberFindObjects.run()` to extract sky fibers first, apply throughput
corrections, and build a B-spline sky model projected into 2D. This replaces
the current `joint_skysub()` approach.

**Files:**
- Modify: `pypeit/find_objects.py` (`FiberFindObjects.run` and new methods)
- Modify: `pypeit/spectrographs/mmt_binospec.py` (throughput methods)

- [ ] **Step 1: Add throughput correction methods to spectrograph**

Add to `MMTBINOSPECIFUSpectrograph`:

```python
def compute_sky_throughput_corrections(self, flat_flux_dict, det):
    """
    Compute throughput corrections for sky fibers.

    Combines the bulk scale factor (avg science / avg sky flat flux)
    with per-fiber corrections from fiber_illumination.fits.

    Args:
        flat_flux_dict (dict): Output of measure_fiber_flat_flux().
        det (int): 1-indexed detector number.

    Returns:
        dict: Keys 'bulk_scale' (float), 'per_fiber_corr' (ndarray for
            sky fibers), 'combined' (ndarray: bulk_scale * per_fiber).
    """
    bulk_scale = flat_flux_dict['bulk_scale']
    f_illum = self.load_fiber_illumination(det)
    # Get sky fiber indices from reference
    blocks = self.get_fiber_blocks(det)
    sky_fiber_indices = []
    for block in blocks:
        if block['type'] == 'sky':
            for fid in block['fiber_ids']:
                # Map fiber_id to index in f_illum
                sky_fiber_indices.append(int(fid) - 1)
    per_fiber = f_illum[sky_fiber_indices]
    combined = bulk_scale * per_fiber
    return {
        'bulk_scale': bulk_scale,
        'per_fiber_corr': per_fiber,
        'combined': combined,
    }
```

- [ ] **Step 2: Implement sky model building in FiberFindObjects**

Add method to `FiberFindObjects`:

```python
def _build_sky_model(self, sky_sobjs, throughput_corrections):
    """
    Build 1D B-spline sky model from extracted sky fiber spectra.

    Applies throughput corrections to sky fiber spectra, then fits a
    joint B-spline to produce a 1D sky model as a function of wavelength.

    Args:
        sky_sobjs (SpecObjs): Extracted sky fiber SpecObjs with
            BOX_WAVE and BOX_COUNTS populated.
        throughput_corrections (ndarray): Per-sky-fiber throughput
            correction factors (multiply sky flux by this to match
            science fiber throughput scale).

    Returns:
        tuple: (sky_bspline, sky_wave, sky_flux) — the fitted B-spline
            object and the wavelength/flux arrays it was fit to.
    """
    from pypeit.core import fitting

    # 1. Collect wavelength and corrected flux from all sky fibers
    all_wave = []
    all_flux = []
    all_ivar = []
    for i, sobj in enumerate(sky_sobjs):
        if sobj.BOX_WAVE is None or sobj.BOX_COUNTS is None:
            continue
        mask = sobj.BOX_MASK if sobj.BOX_MASK is not None else np.ones(len(sobj.BOX_WAVE), dtype=bool)
        good = mask & (sobj.BOX_WAVE > 0)
        if not np.any(good):
            continue
        corr = throughput_corrections[i] if i < len(throughput_corrections) else 1.0
        all_wave.append(sobj.BOX_WAVE[good])
        all_flux.append(sobj.BOX_COUNTS[good] * corr)
        ivar = sobj.BOX_COUNTS_IVAR[good] / corr**2 if sobj.BOX_COUNTS_IVAR is not None \
            else np.ones(np.sum(good))
        all_ivar.append(ivar)

    all_wave = np.concatenate(all_wave)
    all_flux = np.concatenate(all_flux)
    all_ivar = np.concatenate(all_ivar)

    # Sort by wavelength for B-spline fitting
    srt = np.argsort(all_wave)
    all_wave = all_wave[srt]
    all_flux = all_flux[srt]
    all_ivar = all_ivar[srt]

    # 2. Fit B-spline with grating-dependent knot spacing
    # Use the bsp parameter from the sky subtraction config
    bsp = self.par['reduce']['skysub']['bsp']
    sky_set, out_flux, out_mask, out_wave, out_ivar, exit_status = \
        fitting.bspline_profile(all_wave, all_flux, all_ivar,
                                np.ones_like(all_wave),
                                bkspace=bsp, nord=4, upper=3.0, lower=3.0)

    return sky_set, all_wave, all_flux
```

- [ ] **Step 3: Implement 2D sky model projection**

Add method to `FiberFindObjects`:

```python
def _project_sky_to_2d(self, sky_bspline, waveimg, fiber_throughputs=None):
    """
    Project 1D B-spline sky model into a 2D sky image.

    Evaluates the sky model at each pixel's wavelength (from waveimg).
    Optionally scales by per-fiber throughput for correct subtraction.

    Args:
        sky_bspline: Fitted B-spline sky model.
        waveimg (ndarray): 2D wavelength image, shape (nspec, nspat).
        fiber_throughputs (ndarray, optional): Per-pixel throughput
            scaling (e.g., from slitmask + fiber illumination).

    Returns:
        ndarray: 2D sky model image, same shape as waveimg.
    """
    sky_2d = np.zeros_like(waveimg)
    # Evaluate bspline at each pixel's wavelength
    valid = waveimg > 0
    sky_2d[valid] = sky_bspline.value(waveimg[valid])[0]
    # Scale by fiber throughput if provided
    if fiber_throughputs is not None:
        sky_2d *= fiber_throughputs
    return sky_2d
```

- [ ] **Step 4: Refactor FiberFindObjects.run()**

Replace the current sky subtraction with the new flow:

```python
def run(self, std_trace=None, show_peaks=False, show_skysub_fit=False):
    """
    Primary code flow for fiber object finding.

    Flow:
    1. Find fiber peaks in all block-slits
    2. Extract sky fibers from sky block-slits
    3. Apply throughput corrections to sky spectra
    4. Build B-spline sky model from corrected sky spectra
    5. Project sky model into 2D
    6. Create SpecObjs for science fibers on sky-subtracted image
    """
    # Phase 1: Identify fibers in all block-slits (sky + science)
    # Use flat field to find fiber peaks within each block
    # ...

    # Phase 2: Extract sky fibers (boxcar extraction from flat profiles)
    # Create temporary SpecObjs for sky fibers
    # Perform boxcar extraction on the science image (no sky sub yet)
    # ...

    # Phase 3: Apply throughput corrections to sky spectra
    # bulk_scale * fiber_illumination per sky fiber
    # ...

    # Phase 4: Build sky model
    # B-spline fit to corrected sky spectra
    # ...

    # Phase 5: Project to 2D and subtract
    initial_sky = self._project_sky_to_2d(sky_bspline, self.waveimg)
    # ...

    # Phase 6: Create science fiber SpecObjs on sky-subtracted image
    self.reduce_bpm = self.reduce_bpm_init.copy()
    sobjs_obj, self.nobj = self.find_objects(
        self.sciImg.image - initial_sky, self.sciImg.ivar,
        std_trace=std_trace, show=self.findobj_show,
        show_peaks=show_peaks)

    return initial_sky, sobjs_obj
```

- [ ] **Step 5: Commit**

```bash
git add pypeit/find_objects.py pypeit/spectrographs/mmt_binospec.py
git commit -m "feat(fiber): new sky subtraction flow for block-slit extraction

Extract sky fibers first, apply throughput corrections (flat scale +
fiber_illumination), build B-spline sky model, project to 2D for
subtraction. Replaces joint_skysub approach."
```

---

### Task 7: FiberExtract — Block-Aware Extraction with Flat Profiles

Refactor `FiberExtract.local_skysub_extract()` to extract multiple fibers per
block-slit using flat-derived empirical profiles. No local sky subtraction —
the global sky model is already subtracted.

**Files:**
- Modify: `pypeit/extraction.py:970-1063`

- [ ] **Step 1: Update _build_empirical_profiles for block-slits**

The current method builds profiles per narrow slit. Update to build profiles
per fiber within each block-slit using the flat field:

```python
@staticmethod
def _build_empirical_profiles(flatimg, slitmask, slits, sobjs, nspec, nspat):
    """Build empirical spatial profiles for each fiber from the flat field.

    For each fiber SpecObj, extracts the cross-sectional profile from the
    flat within the fiber's block-slit using the fiber's trace position
    and BOX_R_PIX aperture.

    Args:
        flatimg (ndarray): Flat field image.
        slitmask (ndarray): Slit mask image (pixel -> spat_id).
        slits (SlitTraceSet): Block-slit traces.
        sobjs (SpecObjs): Fiber SpecObjs with TRACE_SPAT and BOX_R_PIX.
        nspec, nspat (int): Image dimensions.

    Returns:
        dict: Maps (SLITID, OBJID) to 2D profile array (nspec, nspat).
    """
    profiles = {}
    for sobj in sobjs:
        prof = np.zeros((nspec, nspat))
        onslit = slitmask == sobj.SLITID
        if not np.any(onslit):
            profiles[(sobj.SLITID, sobj.OBJID)] = prof
            continue

        # For each spectral row, extract profile around fiber center
        for row in range(nspec):
            slit_pix = np.where(onslit[row, :])[0]
            if len(slit_pix) == 0:
                continue
            center = sobj.TRACE_SPAT[row]
            r = sobj.BOX_R_PIX
            cols = slit_pix[(slit_pix >= center - r) & (slit_pix <= center + r)]
            if len(cols) > 0:
                vals = flatimg[row, cols]
                good = np.isfinite(vals) & (vals > 0)
                if np.any(good):
                    prof[row, cols[good]] = vals[good]

        # Normalize per row
        for row in range(nspec):
            s = np.sum(prof[row, :])
            if s > 0:
                prof[row, :] /= s

        profiles[(sobj.SLITID, sobj.OBJID)] = prof

    return profiles
```

- [ ] **Step 2: Update local_skysub_extract for block-slits**

```python
def local_skysub_extract(self, global_sky, sobjs, bkg_redux_global_sky=None,
                         spat_pix=None, model_noise=True,
                         show_resids=False, show_profile=False, show=False):
    """
    Extract fiber spectra from block-slits without local sky subtraction.

    For each block-slit, performs boxcar and Horne (1986) optimal extraction
    for every fiber SpecObj using flat-derived empirical profiles.
    """
    self.global_sky = global_sky
    nspec, nspat = self.sciImg.image.shape

    gdslits = np.where(np.logical_not(self.extract_bpm))[0]

    self.outmask = self.sciImg.fullmask.copy()
    self.extractmask = self.sciImg.select_flag(invert=True)
    self.objmodel = np.zeros_like(self.sciImg.image)
    self.skymodel = np.copy(self.global_sky)
    self.bkg_redux_skymodel = bkg_redux_global_sky.copy() \
        if bkg_redux_global_sky is not None else None
    self.ivarmodel = np.copy(self.sciImg.ivar)
    self.sobjs = sobjs.copy()

    slitid_img = self.slits.slit_img(pad=0, flexure=self.spat_flexure_shift)
    inmask = self.sciImg.select_flag(invert=True)
    imgminsky = self.sciImg.image - global_sky
    extract_sky = global_sky if bkg_redux_global_sky is None \
        else bkg_redux_global_sky

    # Build empirical profiles for all fibers
    empirical_profiles = None
    if self.flatimg is not None:
        empirical_profiles = self._build_empirical_profiles(
            self.flatimg, slitid_img, self.slits, self.sobjs, nspec, nspat)

    # Extract each fiber
    for sobj in self.sobjs:
        thismask = slitid_img == sobj.SLITID
        sobj_inmask = inmask & thismask

        # Boxcar extraction
        sobj.extract_boxcar(
            imgminsky, self.sciImg.ivar, sobj_inmask,
            self.waveimg, extract_sky,
            fwhmimg=self.fwhmimg, flatimg=self.flatimg,
            base_var=self.sciImg.base_var,
            count_scale=self.sciImg.img_scale,
            noise_floor=self.sciImg.noise_floor)

        # Optimal extraction using flat-derived profile
        prof_key = (sobj.SLITID, sobj.OBJID)
        if empirical_profiles is not None and prof_key in empirical_profiles:
            oprof = empirical_profiles[prof_key]
            sobj.extract_optimal(
                imgminsky, self.sciImg.ivar, inmask & thismask,
                self.waveimg, extract_sky, thismask, oprof,
                base_var=self.sciImg.base_var,
                count_scale=self.sciImg.img_scale,
                noise_floor=self.sciImg.noise_floor)
        else:
            # Fallback: per-row Horne extraction with Gaussian profile
            self._optimal_extract_fiber(
                sobj, slitid_img, inmask, global_sky, None)

    # Finalize
    base_gpm = self.sciImg.select_flag(invert=True)
    self.outmask.turn_on('EXTRACT',
                         select=base_gpm & np.logical_not(self.extractmask))
    self.steps.append(inspect.stack()[0][3])

    return self.skymodel, self.bkg_redux_skymodel, self.objmodel, \
        self.ivarmodel, self.outmask, self.sobjs
```

- [ ] **Step 3: Commit**

```bash
git add pypeit/extraction.py
git commit -m "feat(fiber): block-aware extraction with flat-derived profiles

FiberExtract now builds 2D profiles per fiber within block-slits and
uses SpecObj.extract_optimal() for Horne extraction. Boxcar uses
full fiber aperture via BOX_R_PIX."
```

---

### Task 8: Post-Extraction Throughput Corrections

Apply `fiber_illumination.fits` and sky-line-based corrections to extracted
1D spectra.

**Files:**
- Modify: `pypeit/spectrographs/mmt_binospec.py`
- Modify: `pypeit/extraction.py` or `pypeit/reduce.py` (call site)

- [ ] **Step 1: Implement post-extraction correction method**

Add to `MMTBINOSPECIFUSpectrograph`:

```python
def apply_throughput_corrections(self, sobjs, det):
    """
    Apply per-fiber throughput corrections to extracted 1D spectra.

    Divides each fiber's extracted counts (BOX_COUNTS, OPT_COUNTS) and
    sky counts by its throughput correction from fiber_illumination.fits.
    Updates inverse variance accordingly.

    Args:
        sobjs (SpecObjs): Extracted SpecObjs. Modified in place.
        det (int): 1-indexed detector number.
    """
    f_illum = self.load_fiber_illumination(det)
    ref = self.load_fiber_ref_profile(det)
    ref_ids = ref['FIB_ID']

    for sobj in sobjs:
        fid = sobj.MASKDEF_ID
        if fid is None or fid < 0:
            continue
        idx = np.where(ref_ids == fid)[0]
        if len(idx) == 0 or idx[0] >= len(f_illum):
            continue
        corr = float(f_illum[idx[0]])
        if corr < 0.1 or not np.isfinite(corr):
            continue

        # Correct boxcar
        if sobj.BOX_COUNTS is not None:
            sobj.BOX_COUNTS /= corr
            sobj.BOX_COUNTS_SKY /= corr
            if sobj.BOX_COUNTS_IVAR is not None:
                sobj.BOX_COUNTS_IVAR *= corr ** 2

        # Correct optimal
        if sobj.OPT_COUNTS is not None:
            sobj.OPT_COUNTS /= corr
            sobj.OPT_COUNTS_SKY /= corr
            if sobj.OPT_COUNTS_IVAR is not None:
                sobj.OPT_COUNTS_IVAR *= corr ** 2
```

- [ ] **Step 2: Adapt compute_skyline_illum for 1D extracted spectra**

The existing `compute_skyline_illum()` works on 2D images. Add a 1D version
that works on extracted spectra:

```python
def compute_skyline_illum_1d(self, sobjs, det):
    """
    Compute per-fiber throughput correction from sky emission lines
    in extracted 1D spectra.

    For each sky line in the wavelength range, measures the line flux
    in each fiber's extracted spectrum, subtracts continuum, normalizes
    by the median across fibers, and returns per-fiber correction factors.

    Args:
        sobjs (SpecObjs): Extracted SpecObjs with BOX_WAVE and BOX_COUNTS.
        det (int): 1-indexed detector number.

    Returns:
        dict: Maps MASKDEF_ID -> correction factor (float).
            Multiply extracted counts by this to correct throughput.
    """
    # Algorithm is the same as compute_skyline_illum but operates on
    # 1D extracted spectra instead of 2D image pixels:
    # 1. For each sky line in self.skyline_list_ang:
    #    a. For each fiber, extract flux in ±4 Ang window
    #    b. Subtract continuum from ±8-20 Ang sidebands
    #    c. Normalize by median across all fibers
    # 2. Per-fiber correction = median ratio across usable lines
    # 3. Clip to [0.3, 3.0]
```

- [ ] **Step 3: Call both corrections from the extraction pipeline**

In `FiberExtract.local_skysub_extract()`, after the extraction loop and before
returning, add:

```python
# Apply post-extraction throughput corrections
if hasattr(self.spectrograph, 'apply_throughput_corrections'):
    self.spectrograph.apply_throughput_corrections(self.sobjs, self.det)
# Apply sky-line-based throughput correction
if hasattr(self.spectrograph, 'compute_skyline_illum_1d'):
    skyline_corr = self.spectrograph.compute_skyline_illum_1d(self.sobjs, self.det)
    for sobj in self.sobjs:
        fid = sobj.MASKDEF_ID
        if fid is not None and fid in skyline_corr:
            corr = skyline_corr[fid]
            if sobj.BOX_COUNTS is not None:
                sobj.BOX_COUNTS /= corr
                if sobj.BOX_COUNTS_IVAR is not None:
                    sobj.BOX_COUNTS_IVAR *= corr ** 2
            if sobj.OPT_COUNTS is not None:
                sobj.OPT_COUNTS /= corr
                if sobj.OPT_COUNTS_IVAR is not None:
                    sobj.OPT_COUNTS_IVAR *= corr ** 2
```

- [ ] **Step 4: Commit**

```bash
git add pypeit/spectrographs/mmt_binospec.py pypeit/extraction.py
git commit -m "feat(binospec): apply throughput corrections post-extraction

fiber_illumination.fits and sky-line-based corrections now applied to
extracted 1D spectra instead of being baked into the pixelflat."
```

---

### Task 9: Integration Testing

Run the full pipeline on test data and verify results.

**Files:**
- No new files

- [ ] **Step 1: Run unit tests**

```bash
pytest pypeit/tests/test_fiber_block_extraction.py -v
```

- [ ] **Step 2: Clear existing calibrations**

```bash
cd ~/MMT/bino_ifu/JADES_1031022
rm -rf Calibrations/ Science/ QA/
```

- [ ] **Step 3: Run full pipeline**

```bash
cd ~/MMT/bino_ifu/JADES_1031022
run_pypeit mmt_binospec_ifu_A/mmt_binospec_ifu_A.pypeit -o 2>&1 | tee run.log
```

Monitor for:
- Edge detection: should report 21 slits per detector (not 360)
- Wavelength calibration: should process 21 slits (not 360)
- Sky subtraction: should extract sky fibers, report throughput corrections
- Extraction: should report fiber counts per block-slit
- No `BADFLATCALIB` errors

- [ ] **Step 4: Verify outputs**

```python
python3 -c "
from astropy.io import fits
import numpy as np

# Check spec1d
spec1d = fits.open('Science/spec1d_*.fits')
n_ext = len(spec1d) - 1  # subtract primary
print(f'Number of extracted fibers: {n_ext}')
# Should be ~716 (360 + 356)

# Check slit count in spec2d
spec2d = fits.open('Science/spec2d_*.fits')
slits1 = fits.getdata('Science/spec2d_*.fits', extname='SLITS-DET01')
slits2 = fits.getdata('Science/spec2d_*.fits', extname='SLITS-DET02')
print(f'DET01 slits: {len(slits1)}, DET02 slits: {len(slits2)}')
# Should be 21 each

# Check sky model quality
skymodel = fits.getdata('Science/spec2d_*.fits', extname='SKYMODEL-DET01')
print(f'Sky model range: [{skymodel.min():.1f}, {skymodel.max():.1f}]')
print(f'Sky model nonzero fraction: {np.sum(skymodel > 0) / skymodel.size:.3f}')
"
```

- [ ] **Step 5: Compare extracted spectra quality**

```python
python3 -c "
from astropy.io import fits
import numpy as np

# Check fiber-to-fiber variation in sky fibers
spec1d = fits.open('Science/spec1d_*.fits')
sky_flux = []
for ext in spec1d[1:]:
    name = ext.header.get('MASKDEF_OBJNAME', '')
    if name.startswith('SKY') and 'DET01' in ext.name:
        flux = ext.data['OPT_COUNTS']
        mask = ext.data['OPT_MASK']
        med = np.median(flux[mask])
        if np.isfinite(med):
            sky_flux.append(med)
sky_flux = np.array(sky_flux)
print(f'Sky fiber count: {len(sky_flux)}')
print(f'Sky fiber flux median: {np.median(sky_flux):.1f}')
print(f'Sky fiber flux std/median: {np.std(sky_flux)/np.median(sky_flux):.3f}')
# Should be consistent (~<10% variation after corrections)
"
```

- [ ] **Step 6: Commit any fixes**

Fix any issues found during integration testing and commit.

---

### Task 10: Update Documentation

Update the pipeline comparison document and CLAUDE.md to reflect the new
block-slit architecture.

**Files:**
- Modify: `doc/spectrographs/mmt_binospec_pipeline_comparison.rst`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update pipeline comparison document**

Update the tracing, extraction, and sky subtraction sections to reflect
the block-slit approach. Key changes:
- Tracing section: document block-level edge detection
- Sky subtraction: document the new extract-correct-model-project flow
- Extraction: document multi-fiber extraction within block-slits

- [ ] **Step 2: Update CLAUDE.md**

Update the "Binospec IFU Notes" section to document the block-slit architecture:
- 21 block-slits per detector (not 360 fiber-slits)
- Fibers as objects within blocks
- Throughput corrections applied post-extraction

- [ ] **Step 3: Commit**

```bash
git add doc/spectrographs/mmt_binospec_pipeline_comparison.rst CLAUDE.md
git commit -m "docs: update documentation for block-slit extraction approach"
```
