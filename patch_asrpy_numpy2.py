"""One-time, guarded fix for asrpy's NumPy 2.x scalar-conversion issue.

Run in the same environment that executes CCV_Stef_MA_NumPy_2_4_3.py:
    conda activate neurohoop
    python patch_asrpy_numpy2.py

The script creates a .bak copy before modifying asrpy's installed source.
It does not install packages or change ASR parameters.
"""
from pathlib import Path
import importlib.util
import sys

spec = importlib.util.find_spec('asrpy')
if spec is None or not spec.submodule_search_locations:
    raise SystemExit('asrpy is not installed in this Python environment.')

path = Path(next(iter(spec.submodule_search_locations))) / 'asr_utils.py'
if not path.is_file():
    raise SystemExit(f'Cannot locate asrpy source: {path}')

source = path.read_text(encoding='utf-8')
original = 'max_width = np.diff(quants)'
corrected = 'max_width = np.diff(quants).item()'

if corrected in source:
    print(f'Already patched: {path}')
    sys.exit(0)
if source.count(original) != 1:
    raise SystemExit(
        'The installed asrpy source does not match the expected version. '
        f'No changes made to {path}. Inspect fit_eeg_distribution manually.'
    )

backup = path.with_suffix('.py.bak')
if backup.exists():
    raise SystemExit(f'Backup already exists: {backup}. No changes made; inspect it first.')

backup.write_text(source, encoding='utf-8')
modified = source.replace(original, corrected, 1)
try:
    compile(modified, str(path), 'exec')
    path.write_text(modified, encoding='utf-8')
except Exception:
    path.write_text(source, encoding='utf-8')
    raise

print('Patched asrpy for NumPy 2.x scalar conversion.')
print('Source:', path)
print('Backup:', backup)
print('Restart your Python/Spyder kernel before running the EEG script.')
