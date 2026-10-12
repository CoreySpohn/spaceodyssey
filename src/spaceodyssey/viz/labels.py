"""Display vocabulary, importable without plotting dependencies."""

from typing import NamedTuple


class QuantityDisplay(NamedTuple):
    """An axis label, numerical transform, and axis scale.

    Attributes:
        label: ASCII text or matplotlib mathtext.
        transform: ``exp`` or ``identity`` applied to stored quantities.
        scale: ``log`` or ``linear`` for the displayed axis.
    """

    label: str
    transform: str
    scale: str


QUANTITY_DISPLAY: dict[str, QuantityDisplay] = {
    "ln_a_eeid": QuantityDisplay(r"$a / a_{\rm EEID}$", "exp", "log"),
    "ln_rp_re": QuantityDisplay(r"$R_p$ ($R_\oplus$)", "exp", "log"),
    "ln_rp_sqrt_a_eeid": QuantityDisplay(
        r"$R_p \sqrt{a / a_{\rm EEID}}$ ($R_\oplus$)", "exp", "log"
    ),
    "log10_h2o_vmr": QuantityDisplay(
        r"$\log_{10}$ H$_2$O volume mixing ratio", "identity", "linear"
    ),
}

CHANNEL_LABELS = {
    "y_issued": "issued",
    "y_cert_pos": "certified positive",
    "y_wrong": "wrong among issued",
    "y_neg": "certified negative",
    "y_neg_wrong": "wrong among negative",
}

CLAIM_LABELS = {
    "habitable_zone": "habitable zone",
    "radius_class": "radius class",
    "water": "water",
}
