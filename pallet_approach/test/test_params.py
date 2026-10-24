"""Check that config/params.yaml matches what the nodes actually declare.

A parameter declared in a node but missing from params.yaml silently falls back
to the default compiled into the node, so the YAML stops being the single source
of truth without anything failing. A parameter in the YAML under the wrong node
name is worse: it looks configured and is simply never read. That is exactly how
the `go_nogo` block came to be addressed to a node that does not exist.

This test is a static check -- it reads the node sources and the YAML, so it
needs neither ROS nor a robot.
"""

import io
import math
import pathlib
import re
import tokenize

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
    'watchdog_node.py': 'watchdog',
    'tracker_node.py': 'pallet_tracker',
    'forks_node.py': 'virtual_forks',
    'dashboard_node.py': 'dashboard',
    'lidar_relay_node.py': 'lidar_relay',
}

DECLARE = re.compile(r"""declare_parameter\(\s*['"]([\w.]+)['"]""")
# names declared from a loop over (name, default) tuples. NOT anchored to the
# start of a line: an earlier version was, so `('a', 1), ('b', 2)` on one line
# declared two parameters and this file checked one. Three parameters reached
# the robot undeclared that way and the node died on the first frame with
# ParameterNotDeclaredException('seed').
TUPLE_NAME = re.compile(r"""\(\s*['"]([\w.]+)['"]\s*,""")
# every way a node reads a parameter back
READ = re.compile(
    r"""get_parameter\(\s*['"]([\w.]+)['"]|(?<![\w.])get\(\s*['"]([\w.]+)['"]""")


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


def strip_comments(source: str) -> str:
    """Blank out comments, keeping line structure.

    The scraper reads text, so it cannot tell code from a comment ABOUT code --
    and a comment in `coordinator_node._declare` explaining that the side
    parameters are written out rather than built from an f-string over
    `('short', 'long')` was duly reported as a parameter named `short`. Amusing
    once; a reason to distrust the check if left in.
    """
    lines = source.splitlines(keepends=True)
    try:
        comments = [token for token in
                    tokenize.generate_tokens(io.StringIO(source).readline)
                    if token.type == tokenize.COMMENT]
    except (tokenize.TokenError, IndentationError):
        # A fragment of a file does not always tokenize. Fall back to the raw
        # text rather than silently scanning nothing, which would make every
        # check above pass vacuously.
        return source
    for token in comments:
        row, column = token.start[0] - 1, token.start[1]
        line = lines[row]
        lines[row] = line[:column] + ' ' * len(token.string) + line[token.end[1]:]
    return ''.join(lines)


@pytest.fixture(scope='module')
def declared():
    """{node name: set of declared parameter names}, read from the sources.

    Only the `_declare` method is scanned, and comments are stripped first.
    Scanning the whole file picks up any tuple of strings as a parameter name
    -- the ROI key list in `_params()`, for instance -- and reports phantom
    parameters; so does a comment that mentions one.
    """
    out = {}
    for filename, node_name in NODE_NAMES.items():
        source = (NODES / filename).read_text()
        start = source.index('def _declare')
        end = source.index('\n    def ', start)
        block = strip_comments(source[start:end])
        out[node_name] = set(DECLARE.findall(block)) | set(TUPLE_NAME.findall(block))
    return out


def test_the_scraper_ignores_comments():
    """Guards the fixture above: if comment stripping broke, every phantom
    parameter would come back and the real checks would drown in them."""
    block = "def _declare(self):\n    # ('short', 'long') are the faces\n" \
            "    self.declare_parameter('real.one', 1.0)\n"
    stripped = strip_comments(block)
    assert 'real.one' in set(DECLARE.findall(stripped))
    assert 'short' not in set(TUPLE_NAME.findall(stripped))


@pytest.fixture(scope='module')
def read_back():
    """{node name: set of parameter names the node reads}.

    Only literal names. A name built from an f-string -- `pallet.sides.{side}`
    -- cannot be checked statically and is skipped; those are covered by
    test_every_declared_parameter_is_in_the_yaml from the other direction.
    """
    out = {}
    for filename, node_name in NODE_NAMES.items():
        source = (NODES / filename).read_text()
        out[node_name] = {a or b for a, b in READ.findall(source)}
    return out


def available_to(node_name, params_file):
    """Every parameter the YAML offers this node: its own block plus `/**`."""
    flat = flatten(params_file.get('/**', {}).get('ros__parameters', {}))
    flat.update(flatten(params_file.get(node_name, {}).get('ros__parameters', {})))
    return flat


def test_params_file_parses(params_file):
    assert '/**' in params_file


@pytest.mark.parametrize('node_name', sorted(NODE_NAMES.values()))
def test_every_parameter_read_is_declared(node_name, declared, read_back):
    """The other direction, and the one that kills the node.

    Reading an undeclared parameter raises ParameterNotDeclaredException. In a
    subscription callback that means every frame fails, so the node looks alive
    -- it publishes diagnostics, it counts frames -- while doing nothing at all.
    Measured 2026-10-15 on the robot: `seed`, `cluster.size_tolerance` and
    `face.max_facing_angle` were all read and none declared.
    """
    undeclared = sorted(read_back[node_name] - declared[node_name])
    assert not undeclared, (
        f'{node_name} reads parameters it never declares: {undeclared}')


def test_every_yaml_block_addresses_a_real_node(params_file):
    """The bug this test exists for: a block named after a node that is not one."""
    for block in params_file:
        assert block == '/**' or block in NODE_NAMES.values(), (
            f'params.yaml has a block for {block!r}, which is not a node in this '
            f'package. Its values are silently never read.')


@pytest.mark.parametrize('node_name', sorted(NODE_NAMES.values()))
def test_every_declared_parameter_is_in_the_yaml(node_name, declared, params_file):
    """Otherwise the node's compiled-in default wins and the YAML is not the
    single source of truth for every tunable value."""
    available = available_to(node_name, params_file)
    missing = sorted(name for name in declared[node_name] if name not in available)
    assert not missing, f'{node_name} declares but params.yaml does not set: {missing}'


def test_the_nodes_declare_something_at_all(declared):
    """Guards the regexes: if they stop matching, every test above passes vacuously."""
    for node_name, names in declared.items():
        minimum = 2 if node_name == 'lidar_relay' else 4   # the relay has two
        assert len(names) >= minimum, f'only found {names} for {node_name}; the regex broke'


def test_speed_limits_are_the_lab_limits(params_file):
    """A tripwire: the lab limits are never raised as a side effect."""
    limits = flatten(params_file['approach_controller']['ros__parameters'])
    assert limits['limits.max_linear'] <= 0.20, 'linear speed limit was raised'
    assert limits['limits.max_angular'] <= 0.30, 'angular speed limit was raised'


def test_every_phase_speed_is_inside_the_limits(params_file):
    """D8 gives each phase its own speed, slower the closer the robot gets.
    They are separate numbers from the limits above, so they need their own
    tripwire -- a per-phase speed above the limit would simply be clamped and
    the intent would be silently lost."""
    block = flatten(params_file['approach_controller']['ros__parameters'])
    shared = flatten(params_file['/**']['ros__parameters'])
    ceiling = block['limits.max_linear']
    phases = {'closing.closing_speed': shared['closing.closing_speed'],
              'reverse.speed': shared['reverse.speed'],
              'insert.speed': block['insert.speed']}
    for name, speed in phases.items():
        assert 0.0 < speed <= ceiling, (
            f'{name} is {speed} m/s against a {ceiling} m/s limit')
    assert phases['reverse.speed'] <= phases['closing.closing_speed'], (
        'reversing is the blindest thing the robot does and must not be the '
        'fastest')


def test_the_closing_loop_can_reach_its_own_arrival_tolerance(params_file):
    """The base cannot turn slower than limits.min_angular, so the smallest
    heading correction it will make is min_angular / k_heading. An arrival
    tolerance tighter than that is one the square-up can never satisfy, and the
    robot would turn back and forth at the standoff until the run timed out."""
    block = flatten(params_file['approach_controller']['ros__parameters'])
    shared = flatten(params_file['/**']['ros__parameters'])
    granularity = block['limits.min_angular'] / shared['closing.k_heading']
    assert shared['closing.arrival_yaw'] > granularity, (
        f"closing.arrival_yaw {shared['closing.arrival_yaw']} is inside the "
        f"base's own {granularity:.4f} rad turn granularity")


def test_the_run_up_target_fits_the_pockets(params_file):
    """closing.residual_target is what the run-up is sized to reach. It has to
    be comfortably inside both the room a tine has in a pocket and the
    GO/NO-GO lateral rule, or a run that performs exactly as designed is
    refused at the standoff."""
    shared = flatten(params_file['/**']['ros__parameters'])
    rules = flatten(params_file['coordinator']['ros__parameters'])
    slack = min((side['pocket_width'] - shared['forks.tine_width']) / 2.0
                for side in sides_of(params_file).values())
    target = shared['closing.residual_target']
    assert target < slack / 2.0, (
        f'a {target * 1000:.0f} mm residual target against {slack * 1000:.0f} mm '
        f'of pocket slack leaves no room for the measurement to be wrong')
    assert target < rules['go_nogo.max_lateral_offset'], (
        'the run-up aims for a residual the GO/NO-GO rules would refuse')


def test_the_minimum_run_up_is_outside_the_fork_tips(params_file):
    """2.5 m is a long way outside, but the check costs nothing and the
    alternative is a run-up that starts with the tines already in the pallet."""
    shared = flatten(params_file['/**']['ros__parameters'])
    reach = shared['forks.carriage_x'] + shared['forks.tine_length']
    assert shared['closing.min_run_up'] > reach
    assert shared['closing.min_run_up'] > shared['standoff_distance']


def test_the_forks_are_mounted_at_the_bumper(params_file):
    """D15 says the carriage sits where a real one would bolt on, which is the
    front face of base_link's collision box. Two places now record that
    number, so they have to agree."""
    shared = flatten(params_file['/**']['ros__parameters'])
    assert shared['forks.carriage_x'] == pytest.approx(shared['robot.front_bumper_x'])


def test_the_robot_is_the_size_it_was_measured(params_file):
    """D23, tape against the URDF. The front overhang carries every clearance
    number in D16 and D21, so a change here is a change to all of them."""
    robot = params_file['/**']['ros__parameters']['robot']
    assert robot['length'] == pytest.approx(0.925, abs=0.01)
    assert robot['width'] == pytest.approx(0.6996, abs=0.01)
    assert robot['front_bumper_x'] - robot['rear_bumper_x'] == pytest.approx(
        robot['length'], abs=1e-9), (
        'the bumper positions do not span the measured length')


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

    # One measured constant, one place. It used to live in the fork_overlay
    # block; the tracker and the insertion controller need the same number, and
    # two copies of a measurement is one copy too many.
    shared = params_file['/**']['ros__parameters']
    assert shared['floor_z'] == pytest.approx(floor_z), (
        'the shared floor_z must match perception\'s expected floor')
    assert 'floor_z' not in params_file['fork_overlay']['ros__parameters'], (
        'floor_z is shared now; a second copy under fork_overlay would be the '
        'one the overlay reads and nothing would say they had diverged')


# Camera geometry, measured 2026-10-14 and used by several tests below.
CAMERA_AHEAD = 0.390                  # m ahead of base_footprint
CAMERA_HEIGHT = 0.398                 # m above the floor (tape)
# The bottom row of the image is at atan((719 - cy) / fy) below the optical
# axis, plus the 0.698 deg the camera is really pitched down by. NOT the
# symmetric half-angle atan(360/fy): cy is 369.75, so using 360 describes the
# TOP edge and makes the camera look 7 mm more capable than it is.
CAMERA_DOWN_TAN = math.tan(math.radians(
    math.degrees(math.atan((719 - 369.74566650390625) / 949.6732177734375)) + 0.698))
BUMPER_X = 0.4625                     # front face of base_link's collision box


def sides_of(params_file):
    return params_file['/**']['ros__parameters']['pallet']['sides']


def in_frame_from(height_above_floor):
    """Closest face distance at which a feature this high is still in frame."""
    return CAMERA_AHEAD + (CAMERA_HEIGHT - height_above_floor) / CAMERA_DOWN_TAN


def test_standoff_clears_the_virtual_fork_tips(params_file):
    """The tines must start OUTSIDE the pallet, or the run begins with the
    forks already through the face.

    This is what sets the standoff (D16). It replaces a test that required the
    pocket opening's bottom lip to stay in frame -- an EPAL feature the
    measured pallet does not have on its pocket face (D4).
    """
    shared = flatten(params_file['/**']['ros__parameters'])
    standoff = shared['standoff_distance']
    reach = shared['forks.carriage_x'] + shared['forks.tine_length']
    assert standoff > reach, (
        f'standoff {standoff} m is inside the fork tip reach {reach:.4f} m: the '
        f'virtual tines would already be in the pallet before insertion starts')
    assert standoff - reach >= 0.05, (
        f'only {(standoff - reach) * 1000:.0f} mm between the tips and the face '
        f'at the standoff')


def test_standoff_keeps_the_pocket_band_in_frame(params_file):
    """Perception has to be able to MEASURE the pockets at the standoff, which
    is a stricter requirement than seeing the top edge of the opening.

    Measured on pallet_s_071 (face at 1.172 m): the face was found to +10 mm
    and zero pockets, because the detection band sits lower than the opening's
    top edge and had already left the frame.
    """
    standoff = params_file['/**']['ros__parameters']['standoff_distance']
    for name, side in sides_of(params_file).items():
        band_bottom = side['band_center'] - side['band_half_height']
        needed = in_frame_from(band_bottom)
        assert standoff >= needed, (
            f'at {standoff} m the {name} side band bottom ({band_bottom:.3f} m '
            f'above the floor) is out of frame; needs {needed:.3f} m')


def test_one_fork_setting_fits_both_faces(params_file):
    """D27: the forks keep ONE spacing and height whichever face is entered.

    That is only allowed because a single setting genuinely fits both, and the
    windows are narrow enough that changing any pocket dimension could close
    them. So this works out the window from the measured geometry rather than
    checking the configured numbers against themselves -- if the long face's
    bottom board grew to 45 mm, the height window would vanish and this fails.
    """
    shared = flatten(params_file['/**']['ros__parameters'])
    width = shared['forks.tine_width']
    thickness = shared['forks.tine_thickness']
    spacing = shared['forks.spacing']
    underside = shared['forks.bottom_height']

    spacing_lo, spacing_hi = 0.0, float('inf')
    height_lo, height_hi = 0.0, float('inf')
    for name, side in sides_of(params_file).items():
        inner = side['pocket_spacing'] / 2 - side['pocket_width'] / 2
        outer = side['pocket_spacing'] / 2 + side['pocket_width'] / 2
        spacing_lo = max(spacing_lo, 2 * (inner + width / 2))
        spacing_hi = min(spacing_hi, 2 * (outer - width / 2))
        height_lo = max(height_lo, side['opening_bottom'])
        height_hi = min(height_hi, side['opening_top'] - thickness)

    assert spacing_lo <= spacing_hi, (
        f'no single fork spacing fits both faces: short needs one window, long '
        f'another, and they do not overlap')
    assert height_lo <= height_hi, 'no single fork height fits both openings'
    assert spacing_lo <= spacing <= spacing_hi, (
        f'forks.spacing {spacing} is outside the {spacing_lo:.3f}-{spacing_hi:.3f} m '
        f'window where one setting fits both faces')
    assert height_lo <= underside <= height_hi, (
        f'forks.bottom_height {underside} is outside the '
        f'{height_lo:.3f}-{height_hi:.3f} m window')


def test_the_forks_are_centred_on_the_face_we_normally_enter(params_file):
    """The short face is the default entry (pallet.entry_face), so the one
    spacing we keep should centre on its pockets. On the long face the tines
    then sit inboard, which is fine and is what the clearance test above
    checks -- but being off-centre on the PRIMARY face would be a choice worth
    arguing for, and nobody has."""
    shared = flatten(params_file['/**']['ros__parameters'])
    expected = params_file['/**']['ros__parameters']['pallet']['entry_face']
    primary = sides_of(params_file)[expected]
    assert shared['forks.spacing'] == pytest.approx(primary['pocket_spacing']), (
        f'forks.spacing {shared["forks.spacing"]} does not match the '
        f'{expected} face\'s pocket spacing {primary["pocket_spacing"]}')


def test_fork_clearance_is_never_marginal(params_file):
    """Worst-case clearance anywhere, so a tightening elsewhere shows up here
    as a number rather than as forks catching on a board."""
    shared = flatten(params_file['/**']['ros__parameters'])
    width, thickness = shared['forks.tine_width'], shared['forks.tine_thickness']
    spacing, underside = shared['forks.spacing'], shared['forks.bottom_height']
    worst = float('inf')
    for name, side in sides_of(params_file).items():
        inner = side['pocket_spacing'] / 2 - side['pocket_width'] / 2
        outer = side['pocket_spacing'] / 2 + side['pocket_width'] / 2
        worst = min(worst,
                    (spacing / 2 - width / 2) - inner,
                    outer - (spacing / 2 + width / 2),
                    underside - side['opening_bottom'],
                    side['opening_top'] - (underside + thickness))
    assert worst >= 0.015, (
        f'worst fork clearance across both faces is {worst * 1000:.0f} mm; '
        f'below 15 mm the pallet\'s own build tolerance eats it')


def test_insertion_stop_keeps_the_tips_inside_the_pallet(params_file):
    """The binding constraint on the long side, and the reason its stop is
    further out than the short side's (D21): that face is only 0.795 m deep, so
    the short side's 0.700 m stop would drive the tips out of the back."""
    shared = flatten(params_file['/**']['ros__parameters'])
    reach = shared['forks.carriage_x'] + shared['forks.tine_length']
    for name, side in sides_of(params_file).items():
        stop = side['insertion_stop_face_x']
        depth = reach - stop
        assert depth < side['depth'], (
            f'{name}: stopping at {stop} m inserts {depth:.4f} m into a '
            f'{side["depth"]} m deep face -- the tips come out of the back')
        assert side['depth'] - depth >= 0.10, (
            f'{name}: tips only {(side["depth"] - depth) * 1000:.0f} mm short '
            f'of the far edge')


def test_insertion_stop_keeps_the_robot_off_the_pallet(params_file):
    """The forks are imaginary; the robot is not. The stop must also sit where
    the LiDAR can still see the face -- it loses it below a 0.204 m bumper gap
    (D13), measured by tools/lidar_coverage.py."""
    for name, side in sides_of(params_file).items():
        gap = side['insertion_stop_face_x'] - BUMPER_X
        assert gap >= 0.204, (
            f'{name}: a {gap:.4f} m bumper gap is inside the LiDAR blind cone, '
            f'so nothing can see the face when the stop fires')


def test_pallet_geometry_is_self_consistent(params_file):
    """Both faces must hang together as faces of one pallet.

    These are tape measurements, not EPAL nominals, so they will not be round
    numbers -- but each pocket still has to fit inside its opening, and the two
    pockets still have to fit across the face with material between and either
    side of them.
    """
    shared = params_file['/**']['ros__parameters']['pallet']
    height = shared['pallet_height']
    margin = flatten(params_file['pallet_perception']['ros__parameters'])['floor.removal_margin']

    for name, side in sides_of(params_file).items():
        opening = side['opening_top'] - side['opening_bottom']
        assert opening > 0, f'{name}: the opening has no height'
        assert side['opening_top'] < height, (
            f'{name}: the opening reaches above the pallet')

        assert side['pocket_spacing'] > side['pocket_width'], (
            f'{name}: the two pockets would overlap')
        assert side['pocket_spacing'] + side['pocket_width'] < side['face_width'], (
            f'{name}: the pockets run off the ends of the face')

        assert side['min_gap_width'] < side['pocket_width'] < side['max_gap_width'], (
            f'{name}: a nominal {side["pocket_width"]} m pocket is outside its '
            f'own accepted gap range '
            f'{side["min_gap_width"]}-{side["max_gap_width"]} m')

        band_low = side['band_center'] - side['band_half_height']
        band_high = side['band_center'] + side['band_half_height']
        assert band_low > margin, (
            f'{name}: the band starts at {band_low:.3f} m, inside the {margin} m '
            f'of floor that remove_floor deletes')
        assert band_high < side['opening_top'], (
            f'{name}: the band reaches up into the board above the opening')


def test_the_two_faces_are_far_enough_apart_to_tell_apart(params_file):
    """D22. Classification is only safe while each side's tolerance window
    excludes the other side's nominal."""
    pallet = params_file['/**']['ros__parameters']['pallet']
    classify = pallet['classify']
    short, long_ = pallet['sides']['short'], pallet['sides']['long']

    separation = abs(long_['pocket_spacing'] - short['pocket_spacing'])
    assert classify['spacing_tolerance'] < separation / 2, (
        f'a {classify["spacing_tolerance"]} m window around each of '
        f'{short["pocket_spacing"]} and {long_["pocket_spacing"]} m overlaps, so '
        f'one measurement could match both sides')

    # the shared probe band must lie inside BOTH openings, or the pass that
    # decides the side cannot see one of them
    low = classify['band_center'] - classify['band_half_height']
    high = classify['band_center'] + classify['band_half_height']
    for name, side in sides_of(params_file).items():
        assert side['opening_bottom'] <= low and high <= side['opening_top'], (
            f'the classification band {low:.3f}-{high:.3f} m is not inside the '
            f'{name} opening {side["opening_bottom"]}-{side["opening_top"]} m')


def test_face_min_points_admits_a_real_face(params_file):
    """Measured on pallet_2m_0deg: the face fit gets 139-146 inliers at 2 m.

    The fit keeps voxels within `ransac_distance` of a vertical plane, and the
    real pallet's front surface is uneven over more than that, so only a couple
    of voxel rows across the face width qualify. Requiring more than one row
    across the full width would reject every real estimate -- 300 did.
    """
    block = flatten(params_file['pallet_perception']['ros__parameters'])
    narrowest = min(side['face_width'] for side in sides_of(params_file).values())
    one_row = narrowest / block['voxel_size']
    assert block['face.min_points'] < 139, (
        f"face.min_points {block['face.min_points']} exceeds the 139 inliers "
        f"a real pallet face produced at 2 m")
    assert block['face.min_points'] >= one_row, (
        'below one voxel row across the face, almost anything vertical passes')


def test_the_slab_is_wider_than_the_face_is_uneven(params_file):
    """Measured on the real pallet: its front surface spans 26 mm from the deck
    boards to the lower blocks, so a slab narrower than that samples only part
    of the face and starves the gap analysis."""
    pallet = flatten(params_file['/**']['ros__parameters'])
    assert pallet['pallet.pocket_slab_half_depth'] * 2 > 0.026




def test_the_overlay_has_a_colour_for_every_coordinator_state(params_file):
    """A join nothing else checks, in both directions.

    `STATE_COLOR_KEYS` lives in `dashboard.py` (moved there 2026-10-19 so
    `dashboard_node`'s HUD and `fork_overlay_node`'s panel share one mapping
    rather than two that can drift apart) and has no ROS import, so this
    imports it directly instead of scraping the source -- more direct than the
    regex this test used while the dict lived inside a node file, and exactly
    as reliable. A state the mapping misses draws grey, which looks
    deliberate; a key it names that params.yaml lacks raises in a callback,
    which does not.

    This is the same shape as the three runtime failures of 2026-10-15 -- a
    parameter declared on one side only, a message field assigned but not
    declared, a tool reading a key that had moved -- and it is checked the
    same way.
    """
    from pallet_approach.coordinator import State
    from pallet_approach.dashboard import STATE_COLOR_KEYS

    missing = sorted(state.value for state in State
                     if state.value not in STATE_COLOR_KEYS)
    assert not missing, (
        f'STATE_COLOR_KEYS has no entry for {missing}, so those states draw '
        f'as whatever the fallback is and look deliberate')

    # Shared (D57): both fork_overlay and dashboard draw in this palette, so
    # it lives in /**, not in either node's own block.
    configured = set(params_file['/**']['ros__parameters']['colors'])
    unknown = sorted(set(STATE_COLOR_KEYS.values()) - configured)
    assert not unknown, (
        f'STATE_COLOR_KEYS maps states to colour keys {unknown}, which '
        f'params.yaml does not define -- whichever node reads it raises on '
        f'the first frame')

    # fork_overlay_node has to be using THIS mapping, not a stale copy of its
    # own -- read statically, because the node imports rclpy.
    source = strip_comments((NODES / 'fork_overlay_node.py').read_text())
    assert re.search(r'from pallet_approach\.dashboard import STATE_COLOR_KEYS',
                     source), (
        'fork_overlay_node no longer imports the shared state-colour mapping; '
        'check it has not gone back to defining its own')


def test_the_controller_builds_its_executor_before_any_subscription_exists():
    """The join this project keeps re-discovering by crashing: a subscription
    is live from the moment it is created, independent of whatever lifecycle
    state the node believes it is in, so a callback can run before the rest of
    `on_configure` has finished setting up what that callback depends on.

    Measured 2026-10-19 on the robot: `on_configure` created the `~/drive`
    subscription, then spent up to two seconds inside
    `_wait_for_a_subscriber`'s spin loop waiting for twist_mux -- during
    which `_on_drive` fired (the coordinator publishes a STOP heartbeat at
    20 Hz even while idle) with `self._executor` still `None`, because that
    assignment was further down the same function. AttributeError, the node's
    process died, and the coordinator's NEXT configure attempt failed too
    because the lifecycle service was no longer there to answer it.

    Read statically because the node imports rclpy and cannot be constructed
    here. This checks the one thing that actually prevents the regression --
    source order -- rather than the defensive None-guard also added in
    _on_drive, which is insurance and not itself what makes the bug not
    happen.
    """
    source = strip_comments((NODES / 'controller_node.py').read_text())
    start = source.index('def on_configure')
    end = source.index('\n    def ', start + 1)
    body = source[start:end]

    executor_at = body.index('self._executor = DriveExecutor(')
    subscription_at = body.index("self.create_subscription(DriveCommand, '~/drive'")
    assert executor_at < subscription_at, (
        'self._executor is constructed AFTER the ~/drive subscription is '
        'created, which reopens the AttributeError measured 2026-10-19: a '
        'message arriving during _wait_for_a_subscriber\'s spin loop would '
        'call _on_drive with self._executor still None')


def test_the_sim_overlay_puts_every_node_on_sim_time(params_file):
    """Every node block in params.yaml pins use_sim_time: false, and a node's
    own block beats /**, so the sim overlay has to name each node again. A node
    left out would run on wall time against Gazebo's stamps -- every age it
    measured would be decades -- and nothing else would say why."""
    overlay = yaml.safe_load((PACKAGE / 'config' / 'sim_overlay.yaml').read_text())
    nodes = {block for block in params_file if block != '/**'}
    on_sim_time = {block for block, body in overlay.items()
                   if block != '/**' and body['ros__parameters'].get('use_sim_time') is True}
    assert on_sim_time == nodes
    assert overlay['/**']['ros__parameters']['use_sim_time'] is True


def test_the_default_overlay_overrides_nothing_a_node_reads(declared):
    overlay = yaml.safe_load((PACKAGE / 'config' / 'no_overlay.yaml').read_text())
    assert list(overlay) == ['/**']
    keys = set(overlay['/**']['ros__parameters'])
    for node_name, names in declared.items():
        assert not keys & set(names), (node_name, keys & set(names))


def test_the_streamer_creates_its_node_before_gstreamer_initialises():
    """video_stream.py calls Gst.init() at import. On amd64 a process with
    NumPy and an initialised GStreamer aborts in rcl when it then creates a
    node (measured 2026-10-20 in the Gazebo sim's stack container). The import
    has to stay inside __init__, after super().__init__."""
    source = strip_comments((NODES / 'video_streamer.py').read_text())
    head = source[:source.index('class VideoStreamerNode')]
    assert 'video_stream' not in head.replace('video_streamer', '')
    init = source[source.index('def __init__'):]
    assert init.index("super().__init__('video_streamer')") < init.index(
        'from pallet_approach.video_stream import VideoStreamer')


def test_the_controller_can_build_its_executor_from_params_yaml(declared, params_file):
    """The configure step, minus ROS: every name executor_params asks for must
    be declared by the controller and set in params.yaml, and the dict it
    builds must be enough to construct a DriveExecutor and run a move with it.

    Found by the Gazebo sim on 2026-10-20: two closing keys were declared and
    set, never passed on, and the controller raised KeyError inside
    on_configure on every run -- invisible, because rclpy's lifecycle does not
    log a callback's exception."""
    from pallet_approach.drive import DriveExecutor, Request, executor_params
    available = available_to('approach_controller', params_file)
    asked = []

    def get(name):
        asked.append(name)
        assert name in declared['approach_controller'], f'{name} is not declared'
        assert name in available, f'{name} is not in params.yaml'
        return available[name]

    executor = DriveExecutor(executor_params(get))
    for mode in ('STOP', 'CLOSE', 'REVERSE', 'INSERT', 'BACKOUT'):
        executor.start(Request(mode))
    assert asked


def test_lifecycle_callbacks_log_their_exceptions():
    """rclpy's lifecycle turns an exception in on_configure into a failed
    transition and logs nothing, so the node that broke stays silent and the
    coordinator reports only that configuring failed. Every lifecycle callback
    in both managed nodes goes through `logged_transition`."""
    for filename in ('controller_node.py', 'perception_node.py'):
        source = strip_comments((NODES / filename).read_text())
        for match in re.finditer(r'\n(\s*)def (on_\w+)\(self, state', source):
            before = source[:match.start()].rstrip().splitlines()[-1].strip()
            assert before == '@logged_transition', (
                f'{filename}: {match.group(2)} is not decorated with @logged_transition')


def test_the_coordinator_never_spins_itself():
    """The coordinator runs in a MultiThreadedExecutor. rclpy.spin_once(self)
    anywhere in it moves the node to the global executor, after which main()'s
    executor spins nothing and the coordinator goes silent mid-run (measured
    2026-10-20 in the Gazebo sim)."""
    source = strip_comments((NODES / 'coordinator_node.py').read_text())
    assert 'MultiThreadedExecutor()' in source
    for forbidden in ('spin_once(', 'spin_until_future_complete(', 'rclpy.spin('):
        assert forbidden not in source, forbidden


def test_tf_waits_sleep_and_listeners_have_their_own_thread():
    """rclpy's lookup_transform(timeout) spins hot, and a listener without its
    own thread cannot deliver the transform being waited for: 81 % of
    fork_overlay's CPU, measured 2026-10-22. Every node goes through
    _common.lookup_waiting, with a threaded listener."""
    for path in NODES.glob('*_node.py'):
        source = strip_comments(path.read_text())
        assert '_buffer.lookup_transform(' not in source, path.name
        # CameraTransformListener always spins its own helper node.
        for call in re.findall(r'(?<!Camera)TransformListener\([^)]*\)', source):
            assert 'spin_thread=True' in call, (path.name, call)


def test_message_payloads_are_assigned_as_array_array():
    """`msg.data = some_bytes` is checked element by element in Python: 58.5 ms
    for one 640x360 image (measured 2026-10-22), where array.array('B', ...)
    takes 1.3 ms."""
    for path in [*NODES.glob('*.py')]:
        source = strip_comments(path.read_text())
        assert not re.search(r'\.data = [\w.]+\.tobytes\(\)', source), path.name
