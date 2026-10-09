from application_settings import APP_PRESETS


def get_preset(application: str):
    key = str(application).lower()
    if key not in APP_PRESETS:
        raise ValueError(f"Unknown application {application!r}; choose bowel, lung, or heart")
    return APP_PRESETS[key]


def event_vocabulary(application: str):
    preset = get_preset(application)
    aliases = preset["label_aliases"]
    labels = []
    for label in preset["target_patterns"]:
        canonical = aliases.get(str(label).upper(), str(label).upper())
        if canonical and canonical not in labels:
            labels.append(canonical)
    return labels
