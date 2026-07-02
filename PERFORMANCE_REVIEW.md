# GPAW performance review — `gpaw.response`

Scope: structural inefficiencies in `gpaw/response/` (and the low-level helpers it
calls) that can plausibly change runtime or peak memory by a *significant fraction*
(> ~20% of a phase). Micro-optimizations are deliberately excluded.

Notation: `nG` = plane waves (10²–10⁴), `nw` = frequencies, `M`/`nt` = transitions
per k-point pair, `nk` = BZ k-points, `nsym` = symmetries, `P` = MPI world size,
`nblocks` = block-communicator size, "1 unit" = one per-rank
`nw_local × nG × nG` complex array (the dominant allocation, often tens of GB).

## Cross-cutting themes

The same handful of structural patterns account for almost all findings:

1. **BLAS-2/rank-1 updates or plain `np.einsum` where a batched BLAS-3 GEMM is
   possible** — the hottest inner loops of tetrahedron chi0, the GW self-energy,
   the new PAW-correction stack, BSE spectrum assembly, and Wannier projection
   all leave 3–50× on the table this way.
2. **Loop-invariant work recomputed inside hot loops** — wavefunction IFFTs,
   PAW tensor setup, symmetry G-maps, XC-kernel grid evaluations and LFC
   construction are recomputed per k-point / per q / per atom / per frequency
   although they depend on much slower-varying indices.
3. **Full-size temporaries of the dominant `wGG` array** — several code paths
   hold 3–5 simultaneous copies where ~2 would do; this directly limits the
   largest treatable system.
4. **Redundant replicated work across MPI ranks** — with the default
   `nblocks=1`, the entire W construction in GW is computed identically on
   every rank; non-TDA BSE diagonalizes serially on rank 0; several large
   arrays are replicated and allreduced instead of distributed.

---

## Tier 1 — highest impact

### 1.1 GW self-energy: per-(n,m) matrix interpolation + BLAS-2 matvecs
`gpaw/response/g0w0.py:830-863`, `gpaw/response/screened_interaction.py:371-403`

For every symmetry op × k-pair × band n × band m (hundreds–thousands of m),
`FullFrequencyHWModel.get_HW` materializes two fresh `(myng × nG)` matrices by
element-wise interpolation between stored Hilbert slices, then does two
matrix-vector products. Everything is bandwidth-bound BLAS-1/2; the
interpolation traffic alone is ~4–5× the matvec flops it feeds.

Since `get_HW` is *linear* in the stored slices `C_{s,w}`, the m-sum can be
reorganized: bin the m-bands by frequency window `w` (with `nbands ≫ nw` each
bin holds many bands), accumulate weighted outer products per bin with one
ZGEMM, and contract once per touched bin against `C_{s,w}` (σ and dσ are both
linear combinations of the same two slices). This turns O(nbands) full-matrix
materializations + matvecs into a handful of compute-bound GEMMs per (n, fxc).

*"Evaluate sigma" is typically the largest G0W0 timer after chi0; expected gain
3–10× on that phase.* (The MPA model is nonlinear in its poles and already in C;
it cannot be factored the same way.)

### 1.2 W construction is only `nblocks`-parallel; default = fully redundant
`gpaw/response/dielectric_calculator.py:17,43-54`,
`gpaw/response/screened_interaction.py:226-231,439-449`

`get_epsinv_wGG` distributes the per-frequency O(nG³) inversions over
`blockcomm` only. With the G0W0 default `nblocks=1` (`g0w0.py:1147`),
**every rank in the world computes every inversion, the full Hilbert-transform
GEMM, and (for MPA) the entire pole fit** — the W phase has parallel speedup
exactly 1 regardless of core count. The machinery for doing better exists:
`distribute_frequencies` (`pw_parallelization.py:246-282`) distributes over
world, and RPA already uses that pattern.

Fix: redistribute chi0 to `Blocks1D(world, nw)` frequency blocks for the
Dyson/Hilbert/MPA step, then redistribute back. *Gain: ×(P/nblocks) on the W
phase — removes the main strong-scaling wall of the W part.*

### 1.3 Γ-point (q→0): 64 full O(nG³) inversions per frequency for a rank-2 update
`gpaw/response/gamma_int.py:21-49`, `gpaw/response/dielectric_calculator.py:127-147`
(found independently by two reviewers)

The Γ-integral averages ε⁻¹ over a 4×4×4 = 64-point grid of offsets
`|q_f| ~ 1e-6`, with a **full `np.linalg.inv` per point per frequency**. But
`chi0_mapping` only replaces row 0, column 0 and element (0,0) of chi0 — a
rank-2 update of one base matrix — and `sqrtV_G` differs materially only at
G=0. The Γ q-point therefore costs 64× the Dyson work of any other q-point;
for Nq = 10–30 IBZ points, 65–85% of *all* inversion work in a GW/BSE run is
this loop, and it repeats for every frequency in full-frequency/MPA GW.

Fix: invert once per frequency, then apply Sherman–Morrison–Woodbury rank-2
updates (O(nG²) each) for the 64 offsets. *Gain: ~30–60× on the Γ-point Dyson
step; typically 3–6× on the total Dyson phase.*

### 1.4 Tetrahedron chi0: one BLAS-2 rank-1 update per transition per frequency
`gpaw/response/integrators.py:504-521` (driven by `integrators.py:470-484`)

`HilbertTetrahedron.run` Python-loops over every transition M and every
frequency in its window, calling `czher` (serial) or a k=1 `mmm` (blocked) —
one memory-bound rank-1 update streaming the whole nG×nG matrix per call, at
10–20% of ZHERK/ZGEMM throughput, plus O(M·⟨nw⟩) Python/scipy call overhead.
The point-integration `Hilbert.run` already demonstrates the right structure
(batch transitions sharing a frequency bin into one GEMM); the weights
`W_Mw`/windows `i0_M,i1_M` needed for bucketing are already available in the
caller. *Gain: 3–10× on the hot kernel of tetrahedron-mode chi0.*

Two more problems in the same integrator (`integrators.py:458-482`):
`deps_tMk` (nspins × M × nk_tesselation float64, easily GB-scale) is computed
**on every rank for the full domain** instead of only the neighbours of local
points; and tetrahedron weights are computed with a per-transition Python loop
into C. *Gain: removes a GB-scale replicated allocation and O(P)-fold
duplicated eigenvalue work.*

### 1.5 New PAW-correction stack (chiks): per-atom `np.einsum` + per-atom Bessel setup
`gpaw/response/matrix_elements.py:341-346`, `gpaw/response/paw.py:336-388,401-421,524-552`

Two independent problems, both regressions relative to the old `pair.py` stack:

- **Application** (`matrix_elements.py:341-346`): per atom, per k-pair,
  `np.einsum('tij,Gij->tG', ...)` without BLAS — `nt·nG·ni²` complex MACs per
  atom at ~10–50× below GEMM throughput. This is exactly one ZGEMM if `F_aGii`
  is stacked once per q into `F_IG` with composite `I=(a,i,i')` and the
  projector outer products are formed as `C_tI`. The old path already uses
  `mmm` here (`pair.py:253`). *Gain: 10–50× on the PAW step, often >30% of
  chiks integrand time for multi-atom systems.*
- **Setup** (`paw.py:524-552`): `calculate_matrix_element_correction` runs per
  **atom** (identical species ⇒ identical `Fbar_Gii`; the position phase is
  applied afterwards), and its radial transform is a dense
  `scipy.special.spherical_jn` evaluation over an `nG × ngrid` mesh per
  `(j1,j2,l,lp)` — the code's own comments flag both ("XXX ... per species",
  "so slow..."). The old stack does the same physics per species with cached
  FFBT splines (`paw.py:54-60,161-260`). Additional redundancy: `Y_G` is
  evaluated inside the m-loops (×(2l1+1)(2l2+1) ≈ up to 25×), and `j_l` is
  recomputed for every (j1,j2,L,lp) though only ~7 distinct l values exist.
  *Gain: nspecies/natoms × 10–100× on per-q setup — minutes → seconds.*
- **Storage**: `Q_aGii`/`F_aGii` are `natoms × nG × ni²` complex
  (`paw.py:511-516,538-551`); ~7 GB for 50 atoms at nG=5000. Storing per
  species and folding the phase into the per-band multiply cuts this by
  natoms/nspecies.

### 1.6 chiks matrix elements: transition-index fold-out of real-space waves
`gpaw/response/matrix_elements.py:123-126,306-318`

`extract_pseudo_waves` IFFTs the unique bands to `ut_hR`, then **copies** them
out to the transition index (`ut1_mytR = ut1_hR[ikpt1.h_myt]` with
`nt ≈ nocc·nbands`), so each occupied band's full-grid wavefunction is
duplicated ~nbands times; the subsequent product and `f_R` multiply allocate
two more `nt_local × Ngrid` complex arrays. Peak ≈ 4 such arrays — e.g. ~4 GB
per rank where ~30 MB of unique data exists. The old `pair.py` path never
materializes this. Fix: keep waves in `h` index and process transitions in
fixed-size chunks (product → ×f_R → FFT → write into `f_mytG[chunk]`); flops
unchanged. *Gain: 10–100× lower peak temporaries — frequently the memory
limiter for large nbands.* The frequency-weight temporaries
`xf_tZg`/`xf_Zgt` (`chiks.py:324,356`) have the same GB-scale
transient problem and the same chunking fix.

### 1.7 BSE direct kernel: factor-nK redundant wavefunction loads and PAW remaps
`gpaw/response/bse.py:665-741,776-803`, `gpaw/response/paw.py:444-467`

Inside the innermost `(iq → iQ → ik1)` loop, `get_k_point` is called for
`kptv1_s`/`kptc1_s`, which depend **only on iK1** — so every valence/conduction
wavefunction is re-IFFT'd once per BZ q-point (redundancy factor nK; each
global K2 is likewise reloaded myKsize times). And `get_density_matrix` calls
`pawcorr.remap_by_symop` (O(natoms·nG·ni²) einsum) per k-pair although it
depends only on (symop, qpd), i.e. only on iQ. For typical BSE band windows
the redundant FFTs are comparable to the useful pair-density FFTs.

Fix: cache k-points per (s, K, band-window) (LRU if memory-bound) and hoist
the pawcorr remap to the iQ level. *Gain: 25–50% of Hamiltonian-build time.*
A further ~2× is available from exploiting hermiticity of the TDA Hamiltonian
(only K2 ≥ K1 blocks need computing).

### 1.8 chi0 q=0: the entire pair-density pass is run twice
`gpaw/response/chi0.py:98-113`, `gpaw/response/pair.py:155-171`

For optical-limit calculations, `Chi0Calculator.calculate()` runs two full
integration passes over the same k-point domain and bands: body, then optical
extension. The optical pass's `get_optical_pair_density` fills the **entire**
`n_nmG` body again (all FFTs, PAW gemms, wavefunction extractions) just to
also get the 3 head columns — duplicating work that is commonly 15–40% of a
q=0 chi0 calculation. Fix: compute head/wings and body from a single
matrix-element evaluation per point (the combined P-index machinery already
exists, `chi0.py:432-443`). *Gain: 15–40% of optical-limit chi0 wall time
(dielectric functions, BSE screening, GW q=0).* Compounding this, the wings
task `HilbertOpticalLimit.run` (`integrators.py:324-347`) is a pure-Python
per-transition loop (M ≈ 10³–10⁵ interpreter iterations per point) that can be
vectorized exactly like `Hilbert.run`.

### 1.9 Non-TDA BSE: serial rank-0 diagonalization and nR·nG² broadcast
`gpaw/response/bse.py:43-65,932-939,1026-1040,1127-1144`

The full H_SS (16·nS² bytes) is gathered onto rank 0, which alone runs
`np.linalg.eig` (O(nS³) serial) plus two more serial O(nS³) ops in
`get_spectral_weights`, while all other ranks idle. `get_chi_wGG` then builds
`C_tGG` (nR·nG² complex — e.g. 72 GB) on rank 0 and **broadcasts the whole
array to every rank**, each of which keeps only its 1/P slice. Fix: keep the
problem distributed (or at minimum scatter slices instead of broadcasting, and
build the spectrum from broadcast eigenvectors block-by-block). *Gain: ×P on
the diagonalization; ×P memory — the difference between "runs" and "OOM" for
non-TDA on dense k-grids.*

---

## Tier 2 — significant, more localized

### 2.1 Default chi0 (point + Hilbert, serial blocks): hermiticity ignored
`gpaw/response/integrators.py:253-259` vs `:97-110`

The serial branch of `Hilbert.run` does two full ZGEMMs per frequency bin for
an update that is exactly Hermitian (real weights) — and the integrator then
*discards half the result*, overwriting one triangle with the conjugate of the
other. The `Hermitian` task in the same file already uses `rk` (ZHERK, half
the flops). Fix: scale rows by `sqrt(p_m)` and use `rk` (handle the rare
negative-weight edge rows separately). *Gain: ~2× on the dominant kernel of
the default chi0 configuration → typically 20–35% of chi0 body time.*

### 2.2 chi0 finalization: 3–4× peak memory and per-chunk re-symmetrization
`gpaw/response/chi0.py:242-273`, `gpaw/response/chi0_data.py:82-101`,
`gpaw/response/pw_parallelization.py:149-205`

`update_chi0_body` allocates a second full `out_WgG`, a third full
`tmp_chi0_wGG` (a *forced* copy in serial), and the BLACS `_redistribute`
allocates a fourth full `outbuf` while the others are alive: peak 3–4 units on
the array that determines the job's memory footprint. Fixes: integrate
directly into `data_WgG` in the fresh-calculate path; add an `out=` parameter
to `_redistribute`/`distribute_as` (BLACS supports caller-provided output);
symmetrize in place in serial. *Gain: peak 3–4× → ~2× — up to ~2× larger
systems per node.* In addition, accumulate-style callers (RPA/fxc ecut
chunking, `gpaw/xc/rpa.py:237-254`) re-run the Hilbert transform, two full
redistributions and the O(nsym·nw·nG²) symmetrization once per band chunk,
although all of these are linear and commute with accumulation — a
`finalize()` step would do them once per q (*10–25% of chunked RPA runs*).

### 2.3 GW `calculate_w`: copy/redistribution chain
`gpaw/response/g0w0.py:1036-1040`, `gpaw/response/pair_functions.py:161-173`,
`gpaw/response/screened_interaction.py:267-268`

Per q and per fxc mode: `copy_with_reduced_pd` allocates a full new chi0 and
does two full BLACS redistributions **even when ecut is unchanged** (the
default); `get_epsinv_wGG` immediately redistributes back (third); `W_wGG` is
allocated `empty_like(einv_wGG)` although it could be computed in place
(ε⁻¹ is only needed per-frequency for the q0 corrections). Peak ≈ 4–5 units vs
an achievable ~2.5–3. *Gain: ~40% peak memory in the W phase (the usual
high-water mark that forces users to raise nblocks) and 2 of 3 full-array
all-to-alls removed.*

### 2.4 df.py Dyson solve: dense GEMM against the identity, no reuse across quantities
`gpaw/response/df.py:101-131,575-590,746-809`

In the most common path (no truncation, RPA) `K_GG = np.eye(nG)` and
`invert_dyson_like_equation` executes a full 2nG³-flop `in_GG @ K_GG` per
frequency before the O(nG³) solve — ~40% pure waste per frequency (a diagonal
K is likewise stored and multiplied dense). And while χ₀ is cached per q,
every call to `get_dielectric_function` / `get_eels_spectrum` /
`get_dynamic_susceptibility` re-runs the full O(nw·nG³) inversion sweep even
though e.g. EELS is derived from the same `InverseDielectricFunction`. Fixes:
special-case identity/diagonal kernels; cache the inverted object per
(q, truncation, xc, direction). *Gain: 1.5–1.7× on the Dyson stage; 2–3× on
common multi-quantity post-processing; ~3× post-processing memory by not
retaining three full wGG buffers (`df.py:198-204`).*

### 2.5 Symmetry G-map construction: O(nsym·nG²) linear search per q
`gpaw/response/symmetrize.py:94-101` (verified directly)

For each symmetry and each G, the target index is found with
`np.argwhere(Q_G == UQ)[0][0]` — a full linear scan with a fresh nG boolean
temporary per G-vector, in a Python loop: nsym·nG² comparisons plus nsym·nG
numpy-call overheads (tens of seconds per q for nG ~ 10⁴), and it runs twice
per chi0 q-point (body + wings build their operators independently,
`chi0.py:270,456`). Fix: `argsort` + `searchsorted` (O(nG log nG) per
symmetry) and share the map between body and wings. *Gain: 100–1000× on map
setup; can dominate the whole symmetrize timer at few frequencies.* The same
pattern at `pair_functions.py:353-372` (`get_inverted_pw_mapping`,
an O(nG²) pure-Python double loop) needs a dict/lexsort lookup.

### 2.6 MPA residue fit materializes the full design matrix
`gpaw/response/mpa_interpolation.py:27-41`

`fit_residue` allocates `A_GGwp` = `2·npols²` full (myng×nG) matrices (npols=8
→ 8× the entire ε⁻¹ input, e.g. 8.2 GB for nG=2000 serial) plus `XTX_GGpp`.
The GG-work *is* vectorized — the memory is the price paid. Fix: accumulate
the normal equations streaming over w (`XTX += conj(A_p(w))⊗A_o(w)`,
`rhs += conj(A_p(w))·X(w)`). *Gain: ~2.5× peak memory of the MPA solve, which
currently exceeds the entire W storage for npols ≥ 4.*

### 2.7 GW ecut extrapolation redoes all pair densities per cutoff
`gpaw/response/g0w0.py:973-1008`

The loop nesting `ie → symop → k-pair → n` reloads and re-IFFTs all bands and
recomputes all pair densities per cutoff, although the reduced-basis `n_mG` is
exactly a G-subset of the largest-ecut one (chi0 itself is already reused
incrementally across ie — that part is well done). Fix: hoist `ie` to the
innermost level and slice `n_mG[:, G2_G1]`. *Gain: up to ~2× on the σ phase of
extrapolated runs.*

### 2.8 BSE spectrum assembly and indirect kernel: non-BLAS einsums, replication
`gpaw/response/bse.py:761-774,1060-1075,493-499,586-594`

- `np.einsum('tw,tAB->wAB', ...)` without `optimize=` runs a genuine
  (nw×nt)@(nt×nG²) contraction through naive C loops instead of ZGEMM
  (5–20×), and every rank materializes χ for **all** frequencies followed by a
  full-array allreduce instead of computing per-frequency blocks with
  reduce-scatter (×P memory/traffic; directly relevant to BSEPlus).
- `add_indirect_kernel` performs nK²/P tiny einsums that are one GEMM with
  row reordering, and calls `get_k_point` (full band-window IFFTs) **only to
  read `.K`, which equals its own argument** — pure waste. For RPA-mode BSE
  this loop is the entire Hamiltonian build (*5–50× available*).
- `rhoex_KmmG` (16·nS·nG bytes, e.g. 13 GB) is allocated on every rank,
  filled in disjoint slices, allreduced, and kept for the whole run — a
  ring-pass/one-sided scheme would cut per-rank storage by ×P.

### 2.9 chiks/mft: k-invariant setup inside the k-loop; duplicated extraction
`gpaw/response/matrix_elements.py:314-316,508-513`, `gpaw/response/chiks.py:257-261`

`f_R` (a full-grid LibXC kernel evaluation for the transverse pair potential,
including construction of a fresh `XC` object) is rebuilt **per k-point pair**
though it is k-independent; `SiteMatrixElementCalculator` additionally
reconstructs the spherical-truncation LFC (spline setup) per k-pair though it
depends only on q. And in `SelfEnhancementCalculator`, the two matrix-element
calculators independently redo the identical wavefunction extraction, IFFTs,
symmetry mapping and pair product for the same kptpair — everything except the
final ×f_R and FFT is byte-identical (*~2× on the extraction phase*).

### 2.10 Old chi0 stack: kpt1 extraction replicated over blockcomm; per-band PAW GEMVs
`gpaw/response/pair.py:129-131,189-195`, `gpaw/response/paw.py:469-475`

`get_kpoint_pair` extracts/IFFTs all occupied bands of kpt1 identically on
every block rank (×nblocks duplicated work at nblocks = 8–64), and
`pawcorr.multiply` is called per band (natoms GEMVs per band instead of one
GEMM per atom per (q,k), *3–10× on that setup step*). Also, the ground state
is opened with `serial_comm`, so `psit_nG[n]` is a lazy per-band file read
repeated for every (q,k) — an I/O cost growing linearly in Nq that a band-block
cache would remove.

---

## Tier 3 — worthwhile, smaller share of total

- **`localft.py:472-510`**: `spherical_jn` evaluated per (l,m) channel instead
  of per l (~6× compute and ~4× memory on the flagged "Slow step"), and the
  whole plane-wave expansion recomputed per atom instead of per species.
- **`modelinteraction.py:183-187,221-235`**: Wannier projections via
  non-BLAS einsums (rewrite as batched ZGEMMs, 3–10×) and the
  q-only `pawcorr.remap_by_symop` recomputed per k-point.
- **`site_kernels.py:302-316,384-455`, `mft.py:121-126`**: wave vectors built
  via three tiled (nG,nG,3) float arrays (~1 GB transients); special functions
  evaluated on all nG² points though Q = G−G′+q takes only O(nG) distinct
  values; J_ab assembled as nsites²·nG² matvec chains where precomputing
  kernel–vector products gives ~1.5·nsites× fewer flops without ever
  materializing `K_aGG`.
- **`site_paw.py:57-87`**: radial trapz integrals inside the (m1,m2) loops —
  up to 25× redundant; hoist to (j1,j2,L,p).
- **`fxc_kernels.py:184-196`**: G-vector matching via a dense (nQ × ndG)
  distance matrix instead of an integer-coordinate dict — minutes → seconds at
  large ecut.
- **`goldstone.py:231-268`**: every λ iterate of the root search re-solves an
  O(nG³) Dyson equation; eigendecompose Ξ once, then each iterate is O(nG²).
- **`density_kernels.py:128-181`**: Bootstrap kernel — all P ranks duplicate
  the identical 120-iteration SCF of nG³ inversions; compute once, broadcast.
- **`susceptibility.py:281-289,461-472`**: eigenmode lineshapes rebuild a full
  A_wGG only to take v†Av — projecting first is ~nG× cheaper and removes the
  nG² buffer.
- **`q0_correction.py:34-90`** and **`screened_interaction.py:273-275`**:
  frequency-independent setup (Monkhorst-Pack grids, `sqrtV_G` outer product,
  `np.eye`) rebuilt every frequency; hoist (code comments already say so).
- **`screened_interaction.py:298-348`**: `dyson_and_W_new` is dead code with
  broken attribute references — ironically the only consumer of the
  ScaLAPACK-distributed inversion in `wgg.py` that finding 1.2 calls for.
  Either wire it in or delete it.
- **`pair_integrator.py:450-458`, `chi0_base.py:119`, `kpoints.py:145-153`**:
  k-point weights recomputed via KD-tree queries per k although they are
  `len(K_K)` of groups already in hand — one-line fix.
- **`hilbert.py:95-103`**: `GWHilbertTransforms.__call__` allocates an
  unblocked full 2-unit output (unlike the nicely blocked in-place
  `HilbertTransform.__call__`).

## Verified non-issues

- `gpaw/utilities/blas.py` wrappers (`mmm`/`rk`/`czher`) hide no copies; debug
  contiguity checks assert rather than copy.
- The per-frequency Python loop around `np.linalg.inv` is not itself a problem
  (loop overhead ≪ O(nG³) payload); the issues are *redundancy* (1.2, 1.3).
- The FFT grid for pair densities cannot be shrunk to the response cutoff: the
  coarse grid is already at the wavefunction Nyquist limit
  (`gpaw/old/pw/descriptor.py:27-31`), so a smaller product grid would alias.
- W_GG in BSE is now computed per IBZ q on the fly (no all-q storage); TDA
  diagonalization uses ScaLAPACK/ELPA properly; χ₀ is cached across derived
  dielectric quantities; `jdos.py` and `susceptibility.py` batching are fine.
- `Hilbert.run`'s per-bin `.T.copy()` transposes and `GenericUpdate`'s
  per-frequency temporaries are O(m·nG) against O(m·nG²) GEMMs — negligible.

## Summary table (ranked)

| # | Finding | Location | Affected phase | Plausible gain |
|---|---------|----------|----------------|----------------|
| 1.1 | Per-(n,m) interpolation + matvecs → binned GEMMs | g0w0.py:830-863 | GW σ (dominant) | 3–10× phase |
| 1.2 | W build redundant across ranks at nblocks=1 | dielectric_calculator.py:17 | GW/BSE W | ×(P/nblocks) phase |
| 1.3 | 64 Γ-point inversions per frequency (rank-2 update) | gamma_int.py:21-49 | Dyson | 3–6× phase; 30–60× at Γ |
| 1.4 | Rank-1 czher per transition per frequency; replicated deps_tMk | integrators.py:504-521,458-482 | tetrahedron chi0 | 3–10× kernel; GB memory |
| 1.5 | Per-atom einsum PAW apply; per-atom Bessel setup | matrix_elements.py:341; paw.py:336-552 | chiks | 10–50×; setup min→s |
| 1.6 | Transition-index fold-out of real-space waves | matrix_elements.py:123,306 | chiks memory | 10–100× peak temporaries |
| 1.7 | k-point reloads ×nK; PAW remap ×myKsize | bse.py:665-741 | BSE H build | 25–50% (+2× hermiticity) |
| 1.8 | Optical q=0: full duplicate pair-density pass | chi0.py:98-113 | q=0 chi0 | 15–40% wall time |
| 1.9 | Non-TDA: serial eig; nR·nG² broadcast | bse.py:43-65,1027-1144 | non-TDA BSE | ×P compute and memory |
| 2.1 | Hilbert serial branch ignores hermiticity | integrators.py:253-259 | default chi0 | 20–35% chi0 body |
| 2.2 | 3–4 full copies in update_chi0_body; per-chunk symmetrize | chi0.py:242-273 | chi0 memory | 3–4× → 2× peak |
| 2.3 | Identity-ecut copy + triple redistribution; W not in place | g0w0.py:1036-1040 | GW memory | ~40% peak |
| 2.4 | GEMM vs identity kernel; no inversion reuse | df.py:101-131,746-809 | df post-proc | 1.5–3× |
| 2.5 | O(nsym·nG²) argwhere G-map; O(nG²) python map | symmetrize.py:94-101 | per-q setup | s–min per q |
| 2.6 | MPA design-matrix materialization | mpa_interpolation.py:27-41 | MPA memory | ~2.5× peak |
| 2.7 | Extrapolation redoes pair densities | g0w0.py:973-1008 | GW σ (extrap) | ~2× phase |
| 2.8 | Non-BLAS einsums, allreduce replication | bse.py:761-774,1060-1075 | BSE spectrum | 5–50×; ×P memory |
| 2.9 | Per-k f_R/LFC rebuild; duplicated extraction | matrix_elements.py:314,508 | chiks/mft | O(nk) setups; ~2× |
| 2.10 | kpt1 ×nblocks; per-band PAW GEMVs; per-band I/O | pair.py:129-195 | old chi0 | ÷nblocks; 3–10× |
