from pathlib import Path
import sys
package_dir = Path(__file__).resolve().parent.parent / 'Src' / 'Farkle'
__path__ = [str(package_dir)]
if str(package_dir.parent) not in sys.path:
    sys.path.insert(0, str(package_dir.parent))
from Farkle import *
