"""Static checks on the .msg / .action definitions.

These exist because the failures they catch are expensive and far away. A
message definition is not compiled by anything on a laptop -- it is compiled by
`colcon build` inside the container, on the Jetson, four minutes into an image
build, and the error it produces names a CMake file and an emPy template rather
than the line you got wrong.

Measured 2026-10-15: the first real container build failed after 259 s on a
backslash in a *comment* in `PalletTask.action`, written on day 0 and never
exercised until then.
"""

import pathlib
import re

import pytest

INTERFACES = pathlib.Path(__file__).resolve().parents[2] / 'pallet_approach_interfaces'
DEFINITIONS = sorted(
    list(INTERFACES.glob('**/*.msg'))
    + list(INTERFACES.glob('**/*.action'))
    + list(INTERFACES.glob('**/*.srv')))


def test_there_are_definitions_to_check():
    """Guard against the glob silently matching nothing after a move."""
    assert DEFINITIONS, f'no interface definitions found under {INTERFACES}'


@pytest.mark.parametrize('path', DEFINITIONS, ids=lambda p: p.name)
def test_no_backslashes_anywhere(path):
    r"""Not even in a comment.

    `rosidl_adapter` copies comments into the generated IDL, and that text is
    decoded with Python's 'unicodeescape' codec. A shell line-continuation at
    the end of a comment line is then an invalid escape:

        UnicodeDecodeError processing template 'struct.idl.em':
        'unicodeescape' codec can't decode byte 0x5c in position 37

    which fails the whole package. Write usage examples on one line instead.
    """
    offenders = [(n, line) for n, line in enumerate(
        path.read_text().splitlines(), 1) if '\\' in line]
    assert not offenders, (
        f'{path.name} has a backslash, which breaks rosidl_adapter:\n' +
        '\n'.join(f'  line {n}: {line}' for n, line in offenders))


@pytest.mark.parametrize('path', DEFINITIONS, ids=lambda p: p.name)
def test_ascii_only(path):
    """Em dashes and typographic quotes travel through the same decode step as
    the backslash above and are not worth the risk in a file nothing renders.
    Use -- and plain quotes."""
    text = path.read_text()
    bad = {ch for ch in text if ord(ch) > 127}
    assert not bad, (
        f'{path.name} has non-ASCII characters {sorted(bad)}; '
        f'they reach the IDL generator unchanged')


FIELD = re.compile(
    r'^\s*([A-Za-z_][\w/\[\]]*(?:\s*\[\s*\d*\s*\])?)\s+([a-z][a-z0-9_]*)\s*(?:#.*)?$')


def declared_fields(path):
    """Field names in a .msg, ignoring comments, blank lines and constants."""
    names = []
    for line in path.read_text().splitlines():
        line = line.split('#')[0].rstrip()
        if not line or line.startswith('---') or '=' in line:
            continue
        match = FIELD.match(line)
        if match:
            names.append(match.group(2))
    return names


def test_every_pallet_estimate_field_is_filled_in():
    """A field added to the .msg and not to `estimate_to_msg` is published as a
    zero or an empty string forever, which looks like data rather than like a
    bug. This is how `side` would have shipped reading '' on every estimate.
    """
    msg = INTERFACES / 'msg' / 'PalletEstimate.msg'
    conversions = (INTERFACES.parent / 'pallet_approach' / 'pallet_approach'
                   / 'ros_conversions.py').read_text()
    missing = [name for name in declared_fields(msg)
               if name != 'header' and f'msg.{name}' not in conversions]
    assert not missing, (
        f'PalletEstimate.msg declares {missing}, which ros_conversions.py '
        f'never assigns')


def test_definitions_survive_the_real_idl_conversion(tmp_path):
    """The actual thing `colcon build` does, when the tooling is reachable.

    The scans above catch the one failure mode we have hit. This catches all of
    them -- bad types, reserved words, malformed defaults -- by running
    `rosidl_adapter` itself, which is the step that failed on the Jetson.

    It skips unless rosidl_adapter and empy are importable, which on a laptop
    with no ROS they are not. To make it run against a ROS install that is on
    the machine but not on the path:

        PYTHONPATH=/opt/ros/iron/lib/python3.10/site-packages \\
            python3 -m pytest test/test_interfaces.py

    Verified 2026-10-15: with the backslash put back, this reproduces the
    Jetson's error verbatim -- "'unicodeescape' codec can't decode byte 0x5c" --
    in under a second instead of 259.
    """
    adapter = pytest.importorskip(
        'rosidl_adapter.msg', reason='rosidl_adapter not importable here')
    pytest.importorskip('em', reason='empy not installed')
    from rosidl_adapter.action import convert_action_to_idl

    root = INTERFACES.resolve()
    failures = []
    for path in DEFINITIONS:
        relative = path.resolve().relative_to(root)
        convert = (adapter.convert_msg_to_idl if path.suffix == '.msg'
                   else convert_action_to_idl)
        try:
            convert(root, 'pallet_approach_interfaces', relative, tmp_path)
        except Exception as error:                       # noqa: BLE001
            failures.append(f'{relative}: {type(error).__name__}: {error}')
    assert not failures, 'rosidl_adapter rejected:\n  ' + '\n  '.join(failures)


def test_ros_conversions_assigns_no_field_the_msg_lacks():
    """The mirror of the test above, and the one that throws at runtime.

    A ROS 2 Python message class uses __slots__, so assigning a field the .msg
    does not declare raises AttributeError -- inside a subscription callback,
    where it is caught by the node's "a perception bug must not take the node
    down" guard and turns into a log line. Measured 2026-10-15 on the robot:
    `floor_valid`, `floor_normal` and `floor_offset` had been assigned and never
    declared, so no estimate was ever published.

    Checking only one direction is not enough: the declared-but-unassigned case
    publishes a plausible wrong number forever, and the assigned-but-undeclared
    case throws. They fail differently and both need saying.
    """
    msg = INTERFACES / 'msg' / 'PalletEstimate.msg'
    source = (INTERFACES.parent / 'pallet_approach' / 'pallet_approach'
              / 'ros_conversions.py').read_text()
    declared = set(declared_fields(msg))
    # assignments inside estimate_to_msg only; other converters build other types
    start = source.index('def estimate_to_msg')
    end = source.index('\ndef ', start + 1) if '\ndef ' in source[start + 1:] else len(source)
    assigned = set(re.findall(r'\bmsg\.([a-z_][a-z0-9_]*)\s*=', source[start:end]))
    undeclared = sorted(assigned - declared)
    assert not undeclared, (
        f'ros_conversions assigns {undeclared}, which PalletEstimate.msg does '
        f'not declare; a Python message class has __slots__, so that is an '
        f'AttributeError at runtime')


def action_sections(path):
    """A .action is goal / result / feedback, separated by ---."""
    parts = path.read_text().split('\n---')
    names = ('goal', 'result', 'feedback')
    out = {}
    for name, part in zip(names, parts):
        fields = []
        for line in part.splitlines():
            line = line.split('#')[0].rstrip()
            if not line or '=' in line:
                continue
            match = FIELD.match(line)
            if match:
                fields.append(match.group(2))
        out[name] = set(fields)
    return out


def test_the_coordinator_sets_only_action_fields_that_exist():
    """Same join as the message one, on the interface the operator actually
    talks to. An action result field assigned but not declared throws at the
    end of a run -- the most expensive moment to find out, because the robot
    has already driven.
    """
    action = INTERFACES / 'action' / 'PalletTask.action'
    sections = action_sections(action)
    source = (INTERFACES.parent / 'pallet_approach' / 'pallet_approach' / 'nodes'
              / 'coordinator_node.py').read_text()
    problems = []
    for variable, section in (('result', 'result'), ('feedback', 'feedback'),
                              ('goal', 'goal')):
        assigned = set(re.findall(rf'\b{variable}\.([a-z_][a-z0-9_]*)', source))
        # ROS adds these to every action type
        assigned -= {'accepted', 'status', 'request', 'sequence_number'}
        unknown = sorted(assigned - sections[section])
        if unknown:
            problems.append(f'{variable}: {unknown} not in the {section} section')
    assert not problems, (
        'coordinator_node uses PalletTask fields that do not exist:\n  '
        + '\n  '.join(problems))


def test_coordinator_status_assigns_only_declared_fields():
    """The same join as `test_ros_conversions_assigns_no_field_the_msg_lacks`,
    for `CoordinatorStatus` -- the newest message, and so the one most likely
    to have a field renamed on one side and not the other before anyone
    notices. Checked both directions for the reason that test gives: a
    declared-but-unassigned field publishes a plausible wrong value forever
    (NaN or '' here, which at least look like "unset"), while an
    assigned-but-undeclared one throws inside `_tick`, at 20 Hz, which the
    node's exception handling does NOT wrap -- unlike perception's per-frame
    guard, a crash in `_tick` takes the whole coordinator down.
    """
    msg = INTERFACES / 'msg' / 'CoordinatorStatus.msg'
    source = (INTERFACES.parent / 'pallet_approach' / 'pallet_approach' / 'nodes'
              / 'coordinator_node.py').read_text()
    declared = set(declared_fields(msg)) - {'header'}
    start = source.index('def _publish_status')
    end = source.index('\n    def ', start + 1)
    body = source[start:end]
    assigned = set(re.findall(r'\bstatus\.([a-z_][a-z0-9_]*)\s*=', body))
    undeclared = sorted(assigned - declared)
    unassigned = sorted(declared - assigned)
    assert not undeclared, (
        f'_publish_status assigns {undeclared}, which CoordinatorStatus.msg '
        f'does not declare -- AttributeError at 20 Hz')
    assert not unassigned, (
        f'CoordinatorStatus.msg declares {unassigned}, which _publish_status '
        f'never sets -- every message publishes it as a silent default')
