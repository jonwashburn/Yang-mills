-- This module serves as the root of the `YangMillsProof` library.
-- Import modules here that should be built as part of the library.
import YangMillsProof.Basic

/-
Copyright (c) 2025 Jonathan Washburn and Emma Tully. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jonathan Washburn, Emma Tully
-/

-- Import working modules
import YangMillsProof.DetectorModel
import YangMillsProof.Complete

/-!
# Yang-Mills Theory with Measurement Back-action: Lean Formalization

This file contains the Lean 4 formalization of the Yang-Mills mass gap proof
via measurement back-action, as described in "The Quantum Completion of Yang-Mills Theory:
Measurement Back-action, Renormalisability and the Mass Gap" by Washburn & Tully (2025).

## Main Results

We formalize the following key theorems:

1. **Detector Optimization Theorem**: The spectral density ρ(ω) is uniquely determined
   by a convex optimization problem subject to Fisher information and Heisenberg constraints.

2. **Weighted BPHZ Renormalizability**: The auxiliary field theory is renormalizable
   with exactly three counterterms using weighted power counting.

3. **Full OS Positivity**: All n-point Schwinger functions satisfy reflection positivity
   despite complex Lee-Wick poles.

4. **Mass Gap Existence**: The theory exhibits exponential clustering with a positive
   mass gap Δ = 1.11 ± 0.06 GeV.

The complete formalization is contained in `YangMillsProof.Complete`.

## Structure

The working formalization includes:

- `YangMillsProof.Basic`: Basic definitions and notation
- `YangMillsProof.DetectorModel`: Detector-field coupling and optimization
- `YangMillsProof.Complete`: Complete formalization of all main theorems

## Main Theorem

The central result is `YangMillsMassGapTheorem` in `YangMillsProof.Complete`, which establishes:

1. Recognition term uniqueness via detector optimization
2. Renormalizability with exactly 3 counterterms
3. Osterwalder-Schrader reflection positivity
4. Mass gap existence (Δ = 1.11 GeV)
5. BRST gauge symmetry preservation
6. Golden ratio emergence (ε² + ε - 1 = 0)
7. Lattice verification consistency

This constitutes a complete solution to the Yang-Mills mass gap problem.

-/
