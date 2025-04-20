import pytest
from unittest.mock import patch, MagicMock, ANY
import os

# Mock the external dependencies
whisper_mock = MagicMock()
pipeline_mock = MagicMock()

# Mock the modules before importing the target module
modules = {
    'faster_whisper': MagicMock(WhisperModel=whisper_mock),
    'pyannote.audio': MagicMock(Pipeline=pipeline_mock),
    'torch': MagicMock(),
    'huggingface_hub': MagicMock(),
    'whycast_transcribe.config': MagicMock(
        MODEL_CACHE_DIR='mock_cache',
        DEVICE='cpu',
        COMPUTE_TYPE='int8',
        HF_TOKEN='mock_token',
        DIARIZATION_MODEL='pyannote/speaker-diarization-3.1'
    )
}

# Use patch.dict to mock the modules in sys.modules
with patch.dict('sys.modules', modules):
    from whycast_transcribe.model_manager import setup_whisper_model, setup_diarization_pipeline

@pytest.fixture(autouse=True)
def reset_mocks():
    whisper_mock.reset_mock()
    pipeline_mock.reset_mock()
    # Reset the mock Pipeline instance returned by Pipeline.from_pretrained
    pipeline_mock.from_pretrained.return_value = MagicMock()

@patch('os.path.exists', return_value=True) # Assume cache dir exists
@patch('os.makedirs') # Mock makedirs
def test_setup_whisper_model_default(mock_makedirs, mock_exists):
    """Test setting up the Whisper model with default settings."""
    model = setup_whisper_model('base')

    whisper_mock.assert_called_once_with(
        'base',
        device='cpu',
        compute_type='int8',
        download_root='mock_cache/whisper'
    )
    assert model == whisper_mock.return_value
    mock_makedirs.assert_called_once_with('mock_cache/whisper', exist_ok=True)

@patch('os.path.exists', return_value=True)
@patch('os.makedirs')
def test_setup_whisper_model_override(mock_makedirs, mock_exists):
    """Test setting up the Whisper model with overridden settings."""
    # Override config values for this test if needed, or pass directly
    model = setup_whisper_model('large-v3', device='cuda', compute_type='float16')

    whisper_mock.assert_called_once_with(
        'large-v3',
        device='cuda',
        compute_type='float16',
        download_root='mock_cache/whisper'
    )
    assert model == whisper_mock.return_value
    mock_makedirs.assert_called_once_with('mock_cache/whisper', exist_ok=True)

@patch('os.path.exists', return_value=False) # Cache dir does not exist initially
@patch('os.makedirs')
def test_setup_whisper_model_creates_dir(mock_makedirs, mock_exists):
    """Test that the cache directory is created if it doesn't exist."""
    setup_whisper_model('small')
    mock_makedirs.assert_called_once_with('mock_cache/whisper', exist_ok=True)

# --- Diarization Tests ---

@patch('os.path.exists', return_value=True)
@patch('os.makedirs')
@patch('whycast_transcribe.model_manager.HF_TOKEN', 'test_token') # Ensure token is set for test
def test_setup_diarization_pipeline_success(mock_makedirs, mock_exists):
    """Test setting up the diarization pipeline successfully."""
    pipeline = setup_diarization_pipeline()

    pipeline_mock.from_pretrained.assert_called_once_with(
        'pyannote/speaker-diarization-3.1',
        use_auth_token='test_token',
        cache_dir='mock_cache/diarization'
    )
    assert pipeline == pipeline_mock.from_pretrained.return_value
    mock_makedirs.assert_called_once_with('mock_cache/diarization', exist_ok=True)


@patch('os.path.exists', return_value=True)
@patch('os.makedirs')
@patch('whycast_transcribe.model_manager.HF_TOKEN', None) # No token set
def test_setup_diarization_pipeline_no_token(mock_makedirs, mock_exists, caplog):
    """Test diarization setup fails gracefully without a token."""
    pipeline = setup_diarization_pipeline()

    assert pipeline is None
    pipeline_mock.from_pretrained.assert_not_called()
    assert "HF_TOKEN environment variable not set" in caplog.text

@patch('os.path.exists', return_value=True)
@patch('os.makedirs')
@patch('whycast_transcribe.model_manager.HF_TOKEN', 'test_token')
def test_setup_diarization_pipeline_load_error(mock_makedirs, mock_exists, caplog):
    """Test diarization setup handles model loading errors."""
    pipeline_mock.from_pretrained.side_effect = Exception("Model loading failed")
    pipeline = setup_diarization_pipeline()

    assert pipeline is None
    pipeline_mock.from_pretrained.assert_called_once()
    assert "Failed to load diarization pipeline" in caplog.text
    assert "Model loading failed" in caplog.text

@patch('os.path.exists', return_value=False) # Cache dir does not exist initially
@patch('os.makedirs')
@patch('whycast_transcribe.model_manager.HF_TOKEN', 'test_token')
def test_setup_diarization_creates_dir(mock_makedirs, mock_exists):
    """Test that the diarization cache directory is created."""
    setup_diarization_pipeline()
    mock_makedirs.assert_called_once_with('mock_cache/diarization', exist_ok=True)

