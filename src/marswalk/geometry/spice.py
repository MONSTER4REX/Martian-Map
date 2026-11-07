"""Real SPICE geometry via spiceypy + NAIF kernels. UNTESTED ON REAL DATA.

The sandbox has neither spiceypy nor the kernels (MARSWALK_CONTEXT.md §5 lists
MRO/MAVEN/Odyssey/TGO sources). An owner furnishes a kernel meta-file and runs
this on their machine, then feeds the resulting positions (converted into the
local ENU frame used by `geometry.visibility`) to the coverage functions.
"""

from __future__ import annotations


def relay_states(  # pragma: no cover - requires spiceypy + kernels
    meta_kernel: str,
    target: str,
    observer: str,
    frame: str,
    et_times,
):
    """Return target states for a list of ephemeris times. UNTESTED ON REAL DATA.

    Thin wrapper over spiceypy.spkpos; the caller furnishes the meta-kernel and
    converts the body-fixed positions into the local ENU frame before use.
    """
    try:
        import spiceypy as spice
    except ImportError as exc:
        raise RuntimeError(
            "geometry.spice.relay_states requires spiceypy + NAIF kernels, which "
            "are not available in the sandbox. Run on an owner machine; this path "
            "is UNTESTED ON REAL DATA."
        ) from exc

    spice.furnsh(meta_kernel)
    try:
        states = []
        for et in et_times:
            pos, _lt = spice.spkpos(target, et, frame, "LT+S", observer)
            states.append(tuple(pos))
        return states
    finally:
        spice.unload(meta_kernel)
