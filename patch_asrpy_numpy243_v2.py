"""Patch two one-element NumPy array -> scalar issues in asrpy.

Run once in the same environment used by Spyder/your EEG script:
    conda activate neurohoop
    python patch_asrpy_numpy243_v2.py

Does not change your EEG settings, calibration or ASR method.
"""
from pathlib import Path
import importlib.util
import shutil
import sys

spec = importlib.util.find_spec('asrpy')
if spec is None or not spec.submodule_search_locations:
    raise SystemExit('asrpy not found in this Python environment: ' + sys.executable)
path = Path(next(iter(spec.submodule_search_locations))) / 'asr_utils.py'
source = path.read_text(encoding='utf-8')
patches = (
    ('max_width = np.diff(quants)', 'max_width = np.diff(quants).item()'),
    ('alpha = (opt_lu[1] - opt_lu[0]) / np.diff(opt_bounds)',
     'alpha = (opt_lu[1] - opt_lu[0]) / np.diff(opt_bounds).item()'),
)
updated = source
for old, new in patches:
    if new in updated:
        print('Already fixed:', new)
    elif updated.count(old) == 1:
        updated = updated.replace(old, new, 1)
        print('Applying:', new)
    else:
        raise SystemExit('Unexpected asrpy version; cannot safely patch: ' + old)

if updated == source:
    print('Both fixes already present:', path)
    raise SystemExit(0)
compile(updated, str(path), 'exec')
backup = path.with_name('asr_utils_before_numpy243_v2.py.bak')
if backup.exists():
    raise SystemExit('Backup exists, not overwriting: ' + str(backup))
shutil.copy2(path, backup)
try:
    path.write_text(updated, encoding='utf-8')
except Exception:
    shutil.copy2(backup, path)
    raise
print('Patch successful:', path)
print('Backup:', backup)
print('Interpreter:', sys.executable)
print('RESTART the Spyder/IPython kernel before rerunning your EEG script.')
