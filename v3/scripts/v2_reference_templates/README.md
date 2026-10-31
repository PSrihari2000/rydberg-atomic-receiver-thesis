# V2 source references only

These MATLAB files are copied from V2 as readable source references. They are **not V3-ready run scripts**. Their paths, data assumptions, frozen settings, and in some cases test workflow are tied to V2. Review and adapt code into new V3 scripts before running anything.

All six files use the old local AoA draw `theta = 2*pi*rand` with a cosine phase term. V3 uses incident AoA uniformly distributed over `[-pi/2, pi/2]` with the same documented cosine convention. This changes the generated channel distribution.

No V2 `.mat` data, anchor caches, learned weights, checkpoints, validation policies, or result outputs were copied here. Do not use these templates to load V2 results as V3 results.
