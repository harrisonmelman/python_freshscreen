"""Build Neuroglancer scenes and Freshscreen display JSONs.

Modules, roughly in the order data flows through them:
    config         -- S3 prefixes, tool paths and other site constants
    naming         -- parse runno / contrast from Freshscreen filenames; build S3 urls
    contrasts      -- per-contrast defaults (display thresholds, overlay colors)
    geometry       -- nhdr header -> voxel sizes and Neuroglancer source transforms
    display_range  -- sample an n5 volume to pick a default intensity window
    luts           -- segment colors and CIVM stats lookup tables
    shaders        -- Neuroglancer shader source for image layers
    output         -- encode a scene as a Neuroglancer url and write the display JSON

Nothing in this package reads files except display_range, luts and output.
"""
