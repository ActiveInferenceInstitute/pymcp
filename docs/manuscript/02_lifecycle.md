# Lifecycle and measured run records

An observation provides one supported integer outcome per modality. Policy
inference consumes the resulting beliefs and returns the full policy posterior and
native negative expected free energy. The positive EFE field is a sign conversion.
Action selection creates a pending token. Selecting again returns the pending
action without advancing the prior. A matching successful environment step or
explicit external commit consumes the token and advances the predictive prior
once. Resetting the agent restores its initial prior and cancels pending action.
Environment reset concerns physical state and is a separate operation.

Run records separate pre-transition observation, belief and state from successor
fields. `requested_steps` records the plan and `num_steps` the executed count.
Session state retains cumulative runs and is intentionally bounded. An unavailable
or evicted historical record is reported as such. Visualizations derive from real
model and run records, using the actual final trajectory endpoint.
