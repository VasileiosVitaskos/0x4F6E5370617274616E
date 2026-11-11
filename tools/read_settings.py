"""The settings a run actually used, read from what the C wrote.

No tool computes a default or carries a copy of a C constant. The C resolves
every setting (the settings file first, its own defaults for the rest) and
writes the result to <out_dir>/settings.cfg, with its compile-time constants
as '# const NAME = value' lines. Every tool reads the run from there, so a
default changed in the C reaches every tool without touching any of them.

    from read_settings import geometry
    g = geometry("tests/tsb/cfg/tsb_drift.cfg")

The C must have run on that cfg first.
"""

import os
import sys

INTS = (
    "channels",
    "window",
    "stride",
    "components",
    "total_bits",
    "warmup",
    "gs_period",
    "angle_period",
    "ring_window",
    "pretty",
)
FLOATS = ("lambda", "step", "basis_average", "theta_threshold_deg", "freeze_ahead")


def _pairs(path):
    """key = value lines, '#' comments dropped, the C's own rules."""
    out = {}
    with open(path) as f:
        for line in f:
            body = line.split("#", 1)[0].strip()
            if body:
                k, v = body.split("=", 1)
                out[k.strip()] = v.strip()
    return out


def load(cfg_path):
    """Every setting the run used, as typed values, plus the constants.

    Only out_dir is read from the cfg itself, to find the C's file. Stops if
    the C has not run, or if the cfg was edited after it last ran: either way
    the numbers would not describe the run the tools are about to read.
    """
    out_dir = _pairs(cfg_path).get("out_dir", "")
    if not out_dir:
        sys.exit(
            "%s has no out_dir: the C writes the settings it used "
            "there, and the tools read them from there" % cfg_path
        )
    path = os.path.join(out_dir, "settings.cfg")
    if not os.path.exists(path):
        sys.exit("missing %s: run the C on %s first" % (path, cfg_path))
    if os.path.getmtime(path) < os.path.getmtime(cfg_path):
        sys.exit(
            "%s is older than %s: the cfg changed after the C last "
            "ran, run the C again" % (path, cfg_path)
        )

    s = _pairs(path)
    for k in INTS:
        s[k] = int(s[k])
    for k in FLOATS:
        s[k] = float(s[k])
    with open(path) as f:
        for line in f:
            if line.startswith("# const "):
                k, v = line[len("# const ") :].split("=", 1)
                v = v.strip()
                s[k.strip()] = int(v) if v.lstrip("-").isdigit() else float(v)
    for k in ("MAX_ALPHABET", "SPARTAN_EPS"):
        if k not in s:
            sys.exit(
                "%s has no '# const %s' line: it was written by an "
                "older build of the C" % (path, k)
            )
    return s


def geometry(cfg_path):
    """The run's settings under the short names the tools use.

    Renaming only. m is channels times window by the C's own definition
    (config.c), the length of one Mode-1 unfolded window.
    """
    s = load(cfg_path)
    return dict(
        input=s["input"],
        out_dir=s["out_dir"],
        d=s["channels"],
        m_time=s["window"],
        m=s["channels"] * s["window"],
        stride=s["stride"],
        k=s["components"],
        total_bits=s["total_bits"],
        lam=s["lambda"],
        warmup=s["warmup"],
        ring_w=s["ring_window"],
        cross_epoch=s["cross_epoch"],
        max_alphabet=s["MAX_ALPHABET"],
        eps=s["SPARTAN_EPS"],
    )
