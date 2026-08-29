"""Per-task evaluation conventions.

The single most common way to produce uncomparable numbers in this field is to
apply one convention to every task. SR and deraining report Y-channel; denoising
and deblurring report RGB; SR additionally shaves `scale` pixels from each
border. This table is the single source of truth -- eval code reads it, never
hardcodes.
"""

PROTOCOLS = {
    "denoise_gaussian": {"y_channel": False, "crop_border": 0,
                         "test_sets": ["CBSD68", "Kodak24", "Urban100"]},
    "denoise_real":     {"y_channel": False, "crop_border": 0,
                         "test_sets": ["SIDD", "DND"]},
    "deblur":           {"y_channel": False, "crop_border": 0,
                         "test_sets": ["GoPro", "HIDE", "RealBlur_J", "RealBlur_R"]},
    "sr":               {"y_channel": True,  "crop_border": "scale",
                         "test_sets": ["Set5", "Set14", "BSD100", "Urban100", "Manga109"]},
    "derain":           {"y_channel": True,  "crop_border": 0,
                         "test_sets": ["Rain100L", "Rain100H", "Test100", "Test1200"]},
}


# The dataset layer names tasks by degradation family (`denoise`), because that
# is what selects a tail and a task id. The protocol table splits denoising into
# synthetic and real, because they use different test sets. These aliases bridge
# the two vocabularies.
#
# `denoise` resolves to the SYNTHETIC protocol, which is what Phase 1 trains and
# evaluates. Real-noise evaluation (SIDD/DND) must be asked for by its own name
# so that it is always a deliberate choice, never a default.
ALIASES = {
    "denoise": "denoise_gaussian",
    "gaussian_denoise": "denoise_gaussian",
}


def get_protocol(task, scale=1):
    task = ALIASES.get(task, task)
    if task not in PROTOCOLS:
        raise KeyError(
            f"unknown task {task!r}; known: {sorted(PROTOCOLS)} "
            f"(aliases: {sorted(ALIASES)})"
        )
    p = dict(PROTOCOLS[task])
    if p["crop_border"] == "scale":
        p["crop_border"] = scale
    return p
