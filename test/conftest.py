import sys
import types
from pathlib import Path

# Prepend repository root to sys.path so tests import local 'flip' package
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Prevent heavy optional emulator imports (requires torch) during tests
stub_emul = types.ModuleType("flip.covariance.emulators")
sys.modules.setdefault("flip.covariance.emulators", stub_emul)

# Stub problematic ravouxnoanchor25 (bad SciPy import in coefficients)
ra_stub_pkg = types.ModuleType("flip.covariance.ravouxnoanchor25")
ra_stub_terms = types.ModuleType("flip.covariance.ravouxnoanchor25.flip_terms")
ra_stub_coeff = types.ModuleType("flip.covariance.ravouxnoanchor25.coefficients")
sys.modules.setdefault("flip.covariance.ravouxnoanchor25", ra_stub_pkg)
sys.modules.setdefault("flip.covariance.ravouxnoanchor25.flip_terms", ra_stub_terms)
sys.modules.setdefault("flip.covariance.ravouxnoanchor25.coefficients", ra_stub_coeff)
