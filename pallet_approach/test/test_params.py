"""Check that config/params.yaml matches what the nodes actually declare.

A parameter declared in a node but missing from params.yaml silently falls back
to the default compiled into the node, so the YAML stops being the single source
of truth without anything failing. A parameter in the YAML under the wrong node
name is worse: it looks configured and is simply never read. That is exactly how
the `go_nogo` block came to be addressed to a node that does not exist.

This test is a static check -- it reads the node sources and the YAML, so it
needs neither ROS nor a robot.
"""

import math
import pathlib
import re

import pytest
import yaml

PACKAGE = pathlib.Path(__file__).resolve().parent.parent
PARAMS_FILE = PACKAGE / 'config' / 'params.yaml'
NODES = PACKAGE / 'pallet_approach' / 'nodes'

# node source file -> the node name it registers with ROS
NODE_NAMES = {
    'perception_node.py': 'pallet_perception',
    'controller_node.py': 'approach_controller',
    'coordinator_node.py': 'coordinator',
    'fork_overlay_node.py': 'fork_overlay',
}

DECLARE = re.compile(r"""declare_parameter\(\s*['"]([\w.]+)['"]""")
# names declared from a loop over (name, default) tuples
TUPLE_NAME = re.compile(r"""^\s*\(\s*['"]([\w.]+)['"]\s*,""", re.MULTILINE)


def flatten(block, prefix=''):
    """YAML nesting -> the dotted names ROS 2 actually uses."""
    flat = {}
    for key, value in (block or {}).items():
        name = f'{prefix}{key}'
        if isinstance(value, dict):
            flat.update(flatten(value, f'{name}.'))
        else:
            flat[name] = value
    return flat


@pytest.fixture(scope='module')
def params_file():
    return yaml.safe_load(PARAMS_FILE.read_text())


@pytest.fixture(scope='module')
def declared():
    """{node name: set of declared parameter names}, read from the sources.

    Only the `_declare` method is scanned. Scanning the whole file picks up any
    tuple of strings as a parameter name -- the ROI key list in `_params()`, for
    instance -- and reports phantom parameters.
    """
    out = {}
    for filename, node_name in NODE_NAMES.items():
        source = (NODES / filename).read_text()
        start = source.index('def _declare')
        end = source.index('\n    def ', start)
        block = source[start:end]
        out[node_name] = set(DECLARE.findall(block)) | set(TUPLE_NAME.findall(block))
    return out


def available_to(node_name, params_file):
    """Every parameter the YAML offers this node: its own block plus `/**`."""
    flat = flatten(params_file.get('/**', {}).get('ros__parameters', {}))
    flat.update(flatten(params_file.get(node_name, {}).get('ros__parameters', {})))
    return flat


def test_params_file_parses(params_file):
    assert '/**' in params_file


def test_every_yaml_block_addresses_a_real_node(params_file):
    """The bug this test exists for: a block named after a node that is not one."""
    for block in params_file:
        assert block == '/**' or block in NODE_NAMES.values(), (
            f'params.yaml has a block for {block!r}, which is not a node in this '
            f'package. Its values are silently never read.')


@pytest.mark.parametrize('node_name', sorted(NODE_NAMES.values()))
def test_every_declared_parameter_is_in_the_yaml(node_name, declared, params_file):
    """Otherwise the node's compiled-in default wins and the YAML is not the
    single source of truth that CLAUDE.md requires."""
    available = available_to(node_name, params_file)
    missing = sorted(name for name in declared[node_name] if name not in available)
    assert not missing, f'{node_name} declares but params.yaml does not set: {missing}'


def test_the_nodes_declare_something_at_all(declared):
    """Guards the regexes: if they stop matching, every test above passes vacuously."""
    for node_name, names in declared.items():
        assert len(names) > 3, f'only found {names} for {node_name}; the regex broke'


def test_speed_limits_are_the_lab_limits(params_file):
    """A tripwire. CLAUDE.md: never raise these without being asked."""
    limits = flatten(params_file['approach_controller']['ros__parameters'])
    assert limits['limits.max_linear'] <= 0.20, 'linear speed limit was raised'
    assert limits['limits.max_angular'] <= 0.30, 'angular speed limit was raised'


def test_perception_works_at_ground_level(params_file):
    """The floor is at base_footprint, so that is where perception works.

    In base_link every height would be a negative offset from a floor at -0.21,
    and the 2.5 cm ambiguity between base_footprint and the wheel contact patch
    would ride along unnoticed inside it.
    """
    frames = params_file['/**']['ros__parameters']['frames']
    assert frames['ground'] == 'base_footprint'

    # Measured 2026-10-14: the lens is 0.398 m above the floor and the URDF puts
    # it 0.163 m above base_link, so base_link is 0.235 m up and base_footprint
    # (which claims 0.210 m) is 25 mm above the real floor.
    floor_z = params_file['pallet_perception']['ros__parameters']['floor']['expected_z']
    assert floor_z == pytest.approx(-0.025), (
        'the expected floor height must be the measured one, not zero')

    roi = params_file['pallet_perception']['ros__parameters']['roi']
    assert roi['z_min'] < floor_z, (
        'the ROI must reach below the floor so the floor plane stays in for RANSAC')
    assert roi['z_max'] >= 0.3, 'the ROI must reach above a loaded pallet'

    overlay_floor = params_file['fork_overlay']['ros__parameters']['floor_z']
    assert overlay_floor == pytest.approx(floor_z), (
        "the overlay's fallback floor must match perception's expected floor")


def test_standoff_keeps_the_whole_pocket_opening_in_frame(params_file):
    """Derived in docs/SYSTEM.md. Seeing the pocket centre is not enough: the
    rules measure the opening height, so the bottom lip must be in frame."""
    standoff = params_file['coordinator']['ros__parameters']['approach']['standoff_distance']
    shared = flatten(params_file['/**']['ros__parameters'])

    camera_ahead = 0.390
    camera_height_above_floor = 0.398        # tape-measured 2026-10-14
    tan_half_fov = 360.0 / 949.6732177734375
    # the bottom lip of the opening: deck boards split what is left of the
    # pallet height above and below the pocket
    pocket_bottom = (shared['pallet.pallet_height']
                     - shared['pallet.pocket_nominal_height']) / 2

    lowest_visible = camera_height_above_floor - (standoff - camera_ahead) * tan_half_fov
    margin = pocket_bottom - lowest_visible
    assert margin >= 0.015, (
        f'standoff {standoff} m leaves only {margin * 1000:.0f} mm of margin on '
        f'the pocket bottom lip; needs at least 15 mm')


def test_fork_spacing_matches_the_pocket_spacing(params_file):
    """The overlay is only a calibration check if the forks are where a real
    fork carriage would be set."""
    shared = flatten(params_file['/**']['ros__parameters'])
    assert shared['forks.spacing'] == pytest.approx(shared['pallet.pocket_nominal_spacing'])


def test_fork_fits_through_the_pocket(params_file):
    shared = flatten(params_file['/**']['ros__parameters'])
    assert shared['forks.tine_width'] < shared['pallet.pocket_nominal_width']
    assert shared['forks.tine_thickness'] < shared['pallet.pocket_nominal_height']


def test_pallet_geometry_is_self_consistent(params_file):
    """The measured pallet must hang together as a pallet.

    These are tape measurements of the real pallet, not EPAL nominals, so they
    will not be round numbers -- but the pocket still has to fit inside the
    pallet with deck boards above and below it, and the two pockets still have
    to fit across the face with blocks between and either side of them.
    """
    pallet = flatten(params_file['/**']['ros__parameters'])

    height = pallet['pallet.pallet_height']
    opening = pallet['pallet.pocket_nominal_height']
    boards = height - opening
    assert boards > 0.02, 'the pocket opening leaves no deck boards'
    assert opening < height, 'the pocket cannot be taller than the pallet'

    width = pallet['pallet.pallet_width']
    pocket = pallet['pallet.pocket_nominal_width']
    spacing = pallet['pallet.pocket_nominal_spacing']
    # two pockets, centred `spacing` apart, must sit inside the face with
    # material between them and at both ends
    assert spacing > pocket, 'the two pockets would overlap'
    assert spacing + pocket < width, 'the pockets run off the ends of the pallet'

    # the band find_pockets samples must sit inside the opening, clear of the
    # floor-removal margin below and the deck board above
    margin = flatten(params_file['pallet_perception']['ros__parameters'])['floor.removal_margin']
    band_low = pallet['pallet.pocket_center_height'] - pallet['pallet.pocket_band_half_height']
    band_high = pallet['pallet.pocket_center_height'] + pallet['pallet.pocket_band_half_height']
    assert band_low > margin, (
        f'the pocket band starts at {band_low:.3f} m, inside the {margin} m of '
        f'floor that remove_floor deletes')
    assert band_high < height - boards / 2, (
        'the pocket band reaches up into the top deck board')


def test_the_slab_is_wider_than_the_face_is_uneven(params_file):
    """Measured on the real pallet: its front surface spans 26 mm from the deck
    boards to the lower blocks, so a slab narrower than that samples only part
    of the face and starves the gap analysis."""
    pallet = flatten(params_file['/**']['ros__parameters'])
    assert pallet['pallet.pocket_slab_half_depth'] * 2 > 0.026


def test_face_min_points_admits_a_real_face(params_file):
    """Measured on pallet_2m_0deg: the face fit gets 139-146 inliers at 2 m.

    The fit keeps voxels within `ransac_distance` of a vertical plane, and the
    real pallet's front surface is uneven over more than that, so only a couple
    of voxel rows across the face width qualify. Requiring more than one row
    across the full width would reject every real estimate -- 300 did.
    """
    block = flatten(params_file['pallet_perception']['ros__parameters'])
    shared = flatten(params_file['/**']['ros__parameters'])
    one_row = shared['pallet.pallet_width'] / block['voxel_size']
    assert block['face.min_points'] < 139, (
        f"face.min_points {block['face.min_points']} exceeds the 139 inliers "
        f"a real pallet face produced at 2 m")
    assert block['face.min_points'] >= one_row, (
        'below one voxel row across the face, almost anything vertical passes')
