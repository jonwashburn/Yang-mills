import Lake
open Lake DSL

package «YangMillsProof» where
  -- add package configuration options here

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.10.0"

@[default_target]
lean_lib «YangMillsProof» where
  -- add library configuration options here
