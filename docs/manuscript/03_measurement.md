# Numerical and protocol evidence

The VFE helper takes a candidate posterior, prior, observation and likelihood. Its
formula is $D_{\mathrm{KL}}(q\Vert p) - \mathbb{E}_q[\log p(o\mid s)]$. The
related exact Bayes helper computes the evidence from the elementwise joint. A
nonzero but tiny supported likelihood remains nonzero. Unsupported finite results
are not fabricated for zero evidence. The regression suite checks signed entropy,
zero mass, finite distributions and explicit rejection of unsupported settings.

Native transport tests exercise JSON-RPC lifecycle and notification behavior,
SSE channel ownership and cleanup, stdio frames, REST request and result envelopes,
asset content types, and propagation of remote failures. The repository separates
source qualification from the Docker image and installed-artifact runtime.
From a prepared environment, `./run.sh test` exercises its fast source tier and
`./run.sh test-full` selects the full tier.
