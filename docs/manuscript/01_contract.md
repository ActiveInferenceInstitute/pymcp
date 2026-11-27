# Model and interface contract

`ToolRegistry` is the public operation surface. Each entry defines accepted
arguments and dispatches to an owned `PyMDPInterface`. Entry points share schemas
and result semantics. Independent server instances hold separate process-local
state. The native agent and filtering calculations use the declared PyMDP package.

A model provides likelihood families $A[o,s]$, transitions $B[s',s,a]$, priors
$D[s]$ and signed outcome utilities $C[o]$. Optional concentration arrays match the
shapes and support of their corresponding distributions. The admission boundary
checks shape, support, finite values and declared allocation limits before native
construction. Normalization is an explicit operation. The exported model contains
singleton dimensions and the admitted configuration.

Filtering supports the fixed point implementation and single-factor exact mode.
Unsupported smoothing or call-time algorithm changes fail explicitly. Admission
of learning parameters does not expose an update operation.
