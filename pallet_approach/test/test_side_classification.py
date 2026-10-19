"""Telling the pallet's two faces apart.

The two faces are different structures (D19), so the side selects the pocket
nominals, the fork spacing and the insertion stop. Getting it wrong measures a
0.375 m pocket against a 0.230 m nominal, or stops insertion 0.25 m too late.
"""

import numpy as np
import pytest

from pallet_approach import perception as P
from pallet_approach.model import Plane

SIDES = {
    'short': {'face_width': 0.795, 'depth': 1.195, 'pocket_width': 0.230,
              'pocket_spacing': 0.365, 'opening_bottom': 0.000,
              'opening_top': 0.100, 'band_center': 0.055,
              'band_half_height': 0.020, 'min_gap_width': 0.15,
              'max_gap_width': 0.32},
    'long': {'face_width': 1.195, 'depth': 0.795, 'pocket_width': 0.375,
             'pocket_spacing': 0.525, 'opening_bottom': 0.020,
             'opening_top': 0.125, 'band_center': 0.0725,
             'band_half_height': 0.025, 'min_gap_width': 0.26,
             'max_gap_width': 0.50},
}
CLASSIFY = {'enabled': True, 'band_center': 0.060, 'band_half_height': 0.010,
            'min_gap_width': 0.15, 'max_gap_width': 0.50,
            'spacing_tolerance': 0.06, 'width_tolerance': 0.20}


# --------------------------------------------------------------- classify_side

@pytest.mark.parametrize('spacing, width, expected', [
    # measured on the real bags, 2026-10-15
    (0.3619, 0.781, 'short'),       # pallet_s_129
    (0.5774, 1.204, 'long'),        # pallet_l_129
    (0.365, 0.795, 'short'),        # exact nominals
    (0.525, 1.195, 'long'),
])
def test_real_measurements_classify_correctly(spacing, width, expected):
    side, _ = P.classify_side(spacing, width, SIDES, CLASSIFY)
    assert side == expected


def test_disagreement_is_ambiguous_not_a_guess():
    """Two signals pointing at different faces means something is wrong with
    the scene. Picking the better score would turn a coin flip into a decision
    that drives forks at geometry 160 mm out."""
    side, detail = P.classify_side(0.365, 1.195, SIDES, CLASSIFY)
    assert side == P.AMBIGUOUS
    assert 'short' in detail['reason'] and 'long' in detail['reason']


def test_no_pocket_pair_is_ambiguous():
    """Spacing is the deciding signal, so without it there is no decision --
    even though face width alone would happily pick one. Measured on
    pallet_s_071: the face was found to +10 mm with zero pockets, because the
    detection band had left the bottom of the frame."""
    side, detail = P.classify_side(None, 0.795, SIDES, CLASSIFY)
    assert side == P.AMBIGUOUS
    assert 'spacing' in detail['reason']


def test_a_spacing_between_the_two_sides_matches_neither():
    side, _ = P.classify_side(0.445, 0.9, SIDES, CLASSIFY)
    assert side == P.AMBIGUOUS


def test_a_clipped_long_face_is_ambiguous_not_long():
    """A consequence of D22 worth stating explicitly, because it is a refusal
    on a perfectly good pallet.

    Occlusion, the ROI crop or the image edge can cut a 1.195 m face down to
    something that measures like a 0.795 m one. The spacing still says `long`
    and is still right -- but D22 requires both signals to agree, so this comes
    out AMBIGUOUS and the coordinator stops.

    That is the conservative direction, and deliberate: width disagreeing with
    spacing means the face is not fully visible, and measuring pocket geometry
    against nominals on a face you cannot fully see is how a damaged pallet
    passes. Resolving it properly is the tracker's job (stage 2) -- several
    frames from slightly different positions, where a clipped face usually
    stops being clipped.
    """
    side, detail = P.classify_side(0.525, 0.80, SIDES, CLASSIFY)
    assert side == P.AMBIGUOUS
    assert detail['by_spacing'] == 'long'
    assert detail['by_width'] == 'short'


def test_the_windows_cannot_both_match():
    """If the tolerance ever exceeded half the separation, one measurement
    could match both sides and `sole_match` would return None -- ambiguous, so
    safe, but silently so. test_params.py pins the configured value; this pins
    the behaviour."""
    separation = SIDES['long']['pocket_spacing'] - SIDES['short']['pocket_spacing']
    wide = dict(CLASSIFY, spacing_tolerance=separation)
    side, _ = P.classify_side(0.445, 0.9, SIDES, wide)
    assert side == P.AMBIGUOUS


# ------------------------------------------------------------------ flattening

def test_flatten_side_gives_the_stage_names():
    flat = P.flatten_side(SIDES['long'])
    assert flat['pallet_width'] == 1.195
    assert flat['pocket_nominal_spacing'] == 0.525
    assert flat['pocket_nominal_height'] == pytest.approx(0.105)
    assert flat['pocket_opening_bottom'] == 0.020


# ----------------------------------------------------- the opening's lower edge

def test_pocket_height_uses_the_known_opening_floor():
    """The opening's lower edge is not observable from either face -- the short
    side's opening starts at the floor, and the long side's 20 mm board has
    already been deleted by remove_floor. So it comes from the side's geometry,
    and a long-side pocket must not be measured as if it started at the floor.
    """
    floor = Plane(normal=np.array([0.0, 0.0, 1.0]), offset=0.0)
    normal = np.array([-1.0, 0.0, 0.0])
    centroid = np.array([1.75, 0.0, 0.07])

    # A long face: blocks either side of and between two 0.375 m pockets, with
    # the deck board above spanning the whole width. The board matters -- it is
    # what the opening's TOP edge is measured from, and without it in the gap
    # the height comes out undefined rather than wrong.
    ys, zs = [], []
    for y in np.arange(-0.60, 0.601, 0.01):
        in_pocket = abs(abs(y) - 0.2625) < 0.1875
        for z in np.arange(0.03, 0.145, 0.005):
            if in_pocket and z < 0.125:        # the opening: no wood
                continue
            ys.append(y)
            zs.append(z)
    face = np.column_stack([np.full(len(ys), 1.75), ys, zs])

    tall = P.find_pockets(face, normal, centroid, floor, 0.025, 0.26, 0.50,
                          0.0725, bin_width=0.02, min_points_per_bin=1,
                          opening_bottom=0.020)
    flat = P.find_pockets(face, normal, centroid, floor, 0.025, 0.26, 0.50,
                          0.0725, bin_width=0.02, min_points_per_bin=1,
                          opening_bottom=0.0)
    assert tall and flat
    assert tall[0].height == pytest.approx(flat[0].height - 0.020, abs=1e-6), (
        'the opening floor must shift the measured height by exactly its own '
        'value; 20 mm is the difference between the two faces of this pallet')
