# Cluster job scripts (worked examples)

These are the actual SLURM scripts behind the measurements quoted in the docs, published so the
methodology is inspectable rather than asserted. They are **not runnable from a clean checkout**:
they source a private driver tree (`$SCRATCH/scripts/lib/`) that is not part of this repository,
and they assume our partition names, node counts and filesystem layout.

Read them for what was run and how it was measured; adapt rather than execute.
