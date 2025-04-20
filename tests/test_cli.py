import subprocess
import sys
import os

def test_cli_help_output():
    # Call the legacy wrapper script with --help and capture output
    cmd = [sys.executable, os.path.join(os.getcwd(), 'transcribe.py'), '--help']
    result = subprocess.run(cmd, capture_output=True, text=True)
    # Should exit cleanly
    assert result.returncode == 0
    # Help text should include program version and key options (may be in stdout or stderr)
    output = result.stdout + result.stderr
    assert 'WHYcast Transcribe v' in output
    assert '--generate-history' in output
    assert '--force' in output
    assert '--batch' in output

# Additional CLI behavior tests can be added here (e.g., --version, --skip-summary)