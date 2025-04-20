import pytest
import subprocess
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
from typing import Generator, Optional # Add Optional import

# Define the root directory of the project relative to the test file
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
TEST_AUDIO_DIR = PROJECT_ROOT / "tests" / "data" # Assuming test data dir
OUTPUT_DIR = PROJECT_ROOT / "tests" / "output"

# Ensure the source directory is in the Python path for module resolution
# This might be necessary if running pytest from the root vs. tests directory
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Mock expensive operations or external dependencies
@pytest.fixture(autouse=True)
def mock_models():
    # Mock Whisper model setup and transcription
    mock_whisper_model = MagicMock()
    mock_segments = [(MagicMock(start=0, end=5, text="Hello world."), MagicMock(language='en', language_probability=0.9))]
    mock_whisper_model.transcribe.return_value = mock_segments

    # Mock Diarization pipeline setup and processing
    mock_diarization_pipeline = MagicMock()
    mock_diarization_pipeline.return_value = MagicMock() # Mock the result of calling the pipeline

    # Mock the setup functions in model_manager
    with patch('whycast_transcribe.model_manager.setup_whisper_model', return_value=mock_whisper_model) as mock_setup_whisper,\
         patch('whycast_transcribe.model_manager.setup_diarization_pipeline', return_value=mock_diarization_pipeline) as mock_setup_diarization,\
         patch('whycast_transcribe.cli.get_audio_duration', return_value=10.0), \
         patch('whycast_transcribe.postprocess.summarizer.generate_summary', return_value="This is a mock summary."), \
         patch('whycast_transcribe.postprocess.blog_generator.generate_blog', return_value="This is a mock blog post."), \
         patch('whycast_transcribe.postprocess.history_generator.extract_history', return_value="This is a mock history."), \
         patch('whycast_transcribe.postprocess.speaker_assigner.assign_speakers', return_value="Speaker 1: Hello world."), \
         patch('whycast_transcribe.postprocess.cleanup.clean_transcript', return_value="Hello world, cleaned."), \
         patch('whycast_transcribe.utils.formatters.convert_to_html', return_value="<html>Mock HTML</html>"), \
         patch('whycast_transcribe.utils.formatters.convert_to_wiki', return_value="== Mock Wiki =="):
        yield mock_setup_whisper, mock_setup_diarization

@pytest.fixture(scope="module")
def setup_test_environment():
    """Create necessary directories for testing."""
    TEST_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Create a dummy audio file for testing path existence
    dummy_audio_path = TEST_AUDIO_DIR / "dummy_audio.mp3"
    dummy_audio_path.touch()

    # Create a dummy transcript file for testing regeneration workflows
    dummy_transcript_path = TEST_AUDIO_DIR / "dummy_transcript.txt"
    with open(dummy_transcript_path, 'w') as f:
        f.write("This is a test transcript.\nIt has multiple lines.\nSpeaker 1: Hello world.")

    yield # Let tests run

    # Teardown: Clean up created files/dirs (optional, depends on strategy)
    # import shutil
    # if OUTPUT_DIR.exists():
    #     shutil.rmtree(OUTPUT_DIR)
    # if dummy_audio_path.exists():
    #     dummy_audio_path.unlink()

def test_cli_basic_transcription(setup_test_environment, mock_models):
    """Test basic transcription workflow via CLI entry point."""
    mock_setup_whisper, mock_setup_diarization = mock_models
    audio_file = str(TEST_AUDIO_DIR / "dummy_audio.mp3")
    output_dir = str(OUTPUT_DIR)

    # Use sys.executable to ensure the correct Python interpreter is used
    command = [
        sys.executable, "-m", "whycast_transcribe.cli",
        "--audio_path", audio_file,
        "--output_dir", output_dir,
        "--model", "tiny", # Model size doesn't matter due to mocking
        "--skip_summary", # Skip summary for simpler test
        "--no-diarize" # Explicitly disable diarization for this test
    ]

    result = subprocess.run(command, capture_output=True, text=True, cwd=PROJECT_ROOT)

    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

    assert result.returncode == 0
    assert "Starting processing for:" in result.stdout or "Starting processing for:" in result.stderr # Check log output
    assert "Transcription complete." in result.stdout or "Transcription complete." in result.stderr

    # Check if whisper model setup was called
    mock_setup_whisper.assert_called_once()
    # Check if diarization setup was NOT called
    mock_setup_diarization.assert_not_called()

    # Check if expected output files were mentioned or potentially created (mocked)
    # Basic check for file mentions in logs/output
    base_filename = os.path.join(output_dir, "dummy_audio")
    assert f"{base_filename}.txt" in result.stdout or f"{base_filename}.txt" in result.stderr
    assert f"{base_filename}_ts.txt" in result.stdout or f"{base_filename}_ts.txt" in result.stderr
    assert f"{base_filename}_cleaned.txt" in result.stdout or f"{base_filename}_cleaned.txt" in result.stderr

def test_cli_with_diarization(setup_test_environment, mock_models):
    """Test transcription with speaker diarization enabled."""
    mock_setup_whisper, mock_setup_diarization = mock_models
    audio_file = str(TEST_AUDIO_DIR / "dummy_audio.mp3")
    output_dir = str(OUTPUT_DIR)

    command = [
        sys.executable, "-m", "whycast_transcribe.cli",
        "--audio_path", audio_file,
        "--output_dir", output_dir,
        "--model", "tiny",
        "--skip_summary",
        "--diarize",  # Enable diarization
        "--min-speakers", "2",
        "--max-speakers", "4"
    ]

    result = subprocess.run(command, capture_output=True, text=True, cwd=PROJECT_ROOT)

    assert result.returncode == 0
    assert "Starting processing for:" in result.stdout or "Starting processing for:" in result.stderr
    assert "Transcription complete." in result.stdout or "Transcription complete." in result.stderr
    assert "Speaker diarization" in result.stdout or "Speaker diarization" in result.stderr

    # Check if both whisper and diarization setup were called
    mock_setup_whisper.assert_called_once()
    mock_setup_diarization.assert_called_once()

    # Check for speaker assignment files
    base_filename = os.path.join(output_dir, "dummy_audio")
    logs = result.stdout + result.stderr
    assert "_speaker_assignment" in logs

def test_cli_summary_generation(setup_test_environment, mock_models):
    """Test summary generation workflow."""
    audio_file = str(TEST_AUDIO_DIR / "dummy_audio.mp3")
    output_dir = str(OUTPUT_DIR)

    command = [
        sys.executable, "-m", "whycast_transcribe.cli",
        "--audio_path", audio_file,
        "--output_dir", output_dir,
        "--model", "tiny",
        "--no-diarize"  # Disable diarization to focus on summary generation
    ]

    result = subprocess.run(command, capture_output=True, text=True, cwd=PROJECT_ROOT)

    assert result.returncode == 0
    assert "Generating summary" in result.stdout or "Generating summary" in result.stderr
    assert "Summary generated" in result.stdout or "Summary generated" in result.stderr
    
    # Check for summary file being mentioned
    base_filename = os.path.join(output_dir, "dummy_audio")
    logs = result.stdout + result.stderr
    assert f"{base_filename}_summary.txt" in logs
    
    # Check for blog generation
    assert "Generating blog" in logs
    assert f"{base_filename}_blog.txt" in logs
    assert f"{base_filename}_blog.html" in logs
    assert f"{base_filename}_blog.wiki" in logs

def test_cli_history_extraction(setup_test_environment, mock_models):
    """Test history extraction workflow."""
    audio_file = str(TEST_AUDIO_DIR / "dummy_audio.mp3")
    output_dir = str(OUTPUT_DIR)

    command = [
        sys.executable, "-m", "whycast_transcribe.cli",
        "--audio_path", audio_file,
        "--output_dir", output_dir,
        "--model", "tiny",
        "--no-diarize",
        "--generate-history"  # Enable history extraction
    ]

    result = subprocess.run(command, capture_output=True, text=True, cwd=PROJECT_ROOT)

    assert result.returncode == 0
    assert "Extracting history" in result.stdout or "Extracting history" in result.stderr
    
    # Check for history files
    base_filename = os.path.join(output_dir, "dummy_audio")
    logs = result.stdout + result.stderr
    assert f"{base_filename}_history.txt" in logs
    assert f"{base_filename}_history.html" in logs
    assert f"{base_filename}_history.wiki" in logs

def test_regenerate_summary(setup_test_environment, mock_models):
    """Test regenerating summary from existing transcript."""
    transcript_file = str(TEST_AUDIO_DIR / "dummy_transcript.txt")
    output_dir = str(OUTPUT_DIR)

    command = [
        sys.executable, "-m", "whycast_transcribe.cli",
        "--regenerate-summary", transcript_file,
        "--output_dir", output_dir
    ]

    result = subprocess.run(command, capture_output=True, text=True, cwd=PROJECT_ROOT)

    assert result.returncode == 0
    assert "Regenerating summary" in result.stdout or "Regenerating summary" in result.stderr
    assert "Generating blog" in result.stdout or "Generating blog" in result.stderr
    
    # Verify no transcription is being performed
    assert "Transcribing" not in result.stdout and "Transcribing" not in result.stderr

def test_regenerate_cleaned(setup_test_environment, mock_models):
    """Test regenerating cleaned transcript from existing raw transcript."""
    transcript_file = str(TEST_AUDIO_DIR / "dummy_transcript.txt")
    output_dir = str(OUTPUT_DIR)

    command = [
        sys.executable, "-m", "whycast_transcribe.cli",
        "--regenerate-cleaned", transcript_file,
        "--output_dir", output_dir
    ]

    result = subprocess.run(command, capture_output=True, text=True, cwd=PROJECT_ROOT)

    assert result.returncode == 0
    assert "Regenerating cleaned transcript" in result.stdout or "Regenerating cleaned transcript" in result.stderr
    
    # Check that cleaning was performed but no transcription
    base_filename = os.path.splitext(os.path.basename(transcript_file))[0]
    logs = result.stdout + result.stderr
    assert f"{base_filename}_cleaned.txt" in logs
    assert "Transcribing" not in logs

def test_regenerate_blog_from_transcript_and_summary(setup_test_environment, mock_models):
    """Test regenerating blog from existing transcript and summary."""
    transcript_file = str(TEST_AUDIO_DIR / "dummy_transcript.txt")
    # Create a mock summary file in the test directory
    summary_file = TEST_AUDIO_DIR / "dummy_transcript_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("This is a test summary.")
    
    output_dir = str(OUTPUT_DIR)

    command = [
        sys.executable, "-m", "whycast_transcribe.cli",
        "--regenerate-blog", transcript_file, str(summary_file),
        "--output_dir", output_dir
    ]

    result = subprocess.run(command, capture_output=True, text=True, cwd=PROJECT_ROOT)

    assert result.returncode == 0
    assert "Regenerating blog" in result.stdout or "Regenerating blog" in result.stderr
    
    # Verify no transcription or summary generation is being performed
    assert "Transcribing" not in result.stdout and "Transcribing" not in result.stderr
    assert "Generating summary" not in result.stdout and "Generating summary" not in result.stderr
    
    # Check for blog files
    base_filename = os.path.join(output_dir, "dummy_transcript")
    logs = result.stdout + result.stderr
    assert f"{base_filename}_blog.txt" in logs
    assert f"{base_filename}_blog.html" in logs
    assert f"{base_filename}_blog.wiki" in logs

def test_batch_processing(setup_test_environment, mock_models):
    """Test batch processing of audio files."""
    # Create multiple dummy audio files
    dummy_files = ["dummy1.mp3", "dummy2.mp3", "dummy3.mp3"]
    for filename in dummy_files:
        file_path = TEST_AUDIO_DIR / filename
        file_path.touch()
    
    output_dir = str(OUTPUT_DIR)
    audio_pattern = str(TEST_AUDIO_DIR / "dummy*.mp3")

    command = [
        sys.executable, "-m", "whycast_transcribe.cli",
        "--batch", audio_pattern,
        "--output_dir", output_dir,
        "--model", "tiny",
        "--no-diarize",
        "--skip-summary"  # Skip for faster test
    ]

    result = subprocess.run(command, capture_output=True, text=True, cwd=PROJECT_ROOT)

    assert result.returncode == 0
    
    # Check that we process multiple files
    logs = result.stdout + result.stderr
    assert "Processing batch" in logs
    for filename in dummy_files:
        assert filename in logs
    
    # Clean up test files
    for filename in dummy_files:
        file_path = TEST_AUDIO_DIR / filename
        if file_path.exists():
            file_path.unlink()

def test_convert_blogs(setup_test_environment, mock_models):
    """Test conversion of blog files to HTML and Wiki formats."""
    # Create a mock blog text file
    blog_dir = OUTPUT_DIR / "blogs"
    blog_dir.mkdir(exist_ok=True)
    blog_file = blog_dir / "test_blog.txt"
    with open(blog_file, 'w') as f:
        f.write("This is a test blog post.")
    
    command = [
        sys.executable, "-m", "whycast_transcribe.cli",
        "--convert-blogs", str(blog_dir)
    ]

    result = subprocess.run(command, capture_output=True, text=True, cwd=PROJECT_ROOT)

    assert result.returncode == 0
    assert "Converting blogs" in result.stdout or "Converting blogs" in result.stderr
    
    # Check for converted files
    logs = result.stdout + result.stderr
    assert "test_blog.html" in logs
    assert "test_blog.wiki" in logs
    
    # Clean up created file
    if blog_file.exists():
        blog_file.unlink()

# Error handling tests
def test_file_not_found_error(setup_test_environment):
    """Test error handling when audio file doesn't exist."""
    non_existent_file = "non_existent_file.mp3"
    
    command = [
        sys.executable, "-m", "whycast_transcribe.cli",
        "--audio_path", non_existent_file
    ]

    result = subprocess.run(command, capture_output=True, text=True, cwd=PROJECT_ROOT)
    
    # Check for appropriate error handling
    assert result.returncode != 0
    assert "Error" in result.stdout or "Error" in result.stderr
    assert "not found" in result.stdout or "not found" in result.stderr or "No such file" in result.stderr
