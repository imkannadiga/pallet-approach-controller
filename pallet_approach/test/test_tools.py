"""A smoke test over the offline tools.

These are not library code, so nothing else covers them -- and that is exactly
how `tools/lidar_coverage.py` came to be broken by a parameter restructure on
2026-10-19 without a single test failing. It read
`pallet.pocket_center_height`, which had moved into `pallet.sides.<face>`, and
the only way to find out was to run it.

This does not test what the tools compute. It checks the two things that break
when something else moves underneath them: that each still parses and imports,
and that each can still find every parameter it reads in the current
`params.yaml`. Both are cheap; neither needs a bag or a robot.
"""

import ast
import importlib.util
import pathlib
import re
import subprocess
import sys

import pytest
import yaml

PACKAGE = pathlib.Path(__file__).resolve().parent.parent
TOOLS = PACKAGE / 'tools'
PARAMS = PACKAGE / 'config' / 'params.yaml'

SCRIPTS = sorted(p for p in TOOLS.glob('*.py') if not p.name.startswith('_'))


def test_there_are_tools_to_check():
    assert SCRIPTS, f'no tools found under {TOOLS}'


@pytest.mark.parametrize('script', SCRIPTS, ids=lambda p: p.name)
def test_parses(script):
    """A syntax error in a tool is otherwise found at the robot."""
    ast.parse(script.read_text())


def imported_names(path):
    names = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names.update(alias.name.split('.')[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split('.')[0])
    return names


def third_party_imports(script):
    """Non-stdlib modules a tool needs, following its local imports.

    Following them matters: `replay.py` imports nothing third-party itself and
    cannot run without `rosbags`, because `_bagutil` needs it. Stopping at the
    first file would have called that tool runnable and then failed on it.
    """
    seen, queue, third_party = set(), [script], set()
    while queue:
        path = queue.pop()
        if path in seen:
            continue
        seen.add(path)
        for name in imported_names(path):
            if name in sys.stdlib_module_names:
                continue
            sibling = TOOLS / f'{name}.py'
            package = PACKAGE / 'pallet_approach' / f'{name}.py'
            if sibling.exists():
                queue.append(sibling)
            elif name == 'pallet_approach':
                continue          # the package itself; its deps are declared
            elif package.exists():
                queue.append(package)
            else:
                third_party.add(name)
    return third_party


@pytest.mark.parametrize('script', SCRIPTS, ids=lambda p: p.name)
def test_help_runs(script):
    """`--help` exercises imports, argparse setup and module-level code.

    Skipped when a dependency is genuinely absent from this interpreter --
    `clock_audit` needs rclpy, which exists only on the robot. The check reads
    the tool's own imports and tries them, rather than matching the wording of
    its "please install" message: those messages are for people, and a test
    that greps them breaks whenever one is reworded.
    """
    missing = sorted(name for name in third_party_imports(script)
                     if importlib.util.find_spec(name) is None)
    if missing:
        pytest.skip(f'{script.name} needs {", ".join(missing)}, '
                    f'not importable by {pathlib.Path(sys.executable).name}')

    result = subprocess.run([sys.executable, str(script), '--help'],
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, (
        f'{script.name} --help failed:\n{result.stderr[-1500:]}')


# `shared['pallet']['x']`, `block['floor']['y']`, and so on: a tool reaching
# into params.yaml by hand rather than through _bagutil.load_params.
DIRECT_READ = re.compile(r"""\[['"]pallet['"]\]((?:\[['"][\w]+['"]\])+)""")


@pytest.mark.parametrize('script', SCRIPTS, ids=lambda p: p.name)
def test_every_pallet_key_a_tool_reads_still_exists(script):
    """The failure that prompted this file.

    A tool that goes straight into the YAML is coupled to its shape, and the
    shape changes. `_bagutil.load_params` exists so most tools do not have to,
    but the ones that do should at least break loudly here rather than on the
    robot.
    """
    pallet = yaml.safe_load(PARAMS.read_text())['/**']['ros__parameters']['pallet']
    missing = []
    for match in DIRECT_READ.finditer(script.read_text()):
        keys = re.findall(r"""['"](\w+)['"]""", match.group(1))
        node = pallet
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                missing.append('pallet.' + '.'.join(keys))
                break
            node = node[key]
    assert not missing, (
        f'{script.name} reads {sorted(set(missing))}, which params.yaml no '
        f'longer has')
