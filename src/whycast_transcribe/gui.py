#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GUI voor WHYcast-transcribe.

Deze module bevat een PyQt5-interface voor het WHYcast-transcribe programma,
waarmee gebruikers audio bestanden kunnen selecteren of RSS feeds kunnen invoeren
en de transcriptie en andere output real-time kunnen bekijken.
"""

import os
import sys
import logging
import feedparser
import tempfile
import random
import requests
import traceback
from datetime import datetime
from typing import Optional, Tuple, List, Dict
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QFileDialog, QLineEdit, QProgressBar,
    QCheckBox, QSpinBox, QComboBox, QMessageBox, QSplitter, QListWidget, QListWidgetItem
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QFont, QTextCursor

from whycast_transcribe.config import (
    VERSION, MODEL_SIZE, USE_SPEAKER_DIARIZATION,
    DIARIZATION_MIN_SPEAKERS, DIARIZATION_MAX_SPEAKERS
)
from whycast_transcribe.model_manager import setup_models
from whycast_transcribe.transcribe import transcribe_audio
from whycast_transcribe.utils.file_helpers import write_transcript_files
from whycast_transcribe.postprocess import run_postprocessing
from whycast_transcribe.audio_processor import detect_audio_format, convert_audio_to_wav


# Aangepaste logger die output naar de GUI stuurt
class QTextEditLogger(logging.Handler):
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget
        self.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        
    def emit(self, record):
        msg = self.format(record)
        self.text_widget.append(msg)
        self.text_widget.moveCursor(QTextCursor.End)


# Worker thread voor het transcriptieproces
class TranscribeWorker(QThread):
    progress = pyqtSignal(str)
    finished_signal = pyqtSignal(tuple)
    error = pyqtSignal(str)
    
    def __init__(self, audio_file, model_size="large-v3", use_diarization=True,
                 min_speakers=None, max_speakers=None):
        super().__init__()
        self.audio_file = audio_file
        self.model_size = model_size
        self.use_diarization = use_diarization
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        
    def run(self):
        try:
            # Modellen instellen
            self.progress.emit(f"Modellen laden voor {self.model_size}...")
            model, diarization_pipeline = setup_models(self.model_size, self.use_diarization)
            
            # Audio transcriberen
            self.progress.emit(f"Transcriptie starten voor {os.path.basename(self.audio_file)}...")
            segments, info, speaker_segments = transcribe_audio(
                model, 
                self.audio_file, 
                diarization_pipeline,
                min_speakers=self.min_speakers,
                max_speakers=self.max_speakers,
                live_callback=self.progress.emit  # Pass signal emit as callback
            )
            
            # Outputbestanden maken
            base = os.path.splitext(self.audio_file)[0]
            transcript_file = f"{base}.txt"
            timestamped_file = f"{base}_ts.txt"
            self.progress.emit(f"Transcriptie opslaan naar {os.path.basename(transcript_file)}...")
            transcript = write_transcript_files(segments, transcript_file, timestamped_file, speaker_segments)
            
            # Postprocessing uitvoeren
            self.progress.emit("Postprocessing starten...")
            run_postprocessing(segments, self.audio_file)
            
            self.progress.emit("Transcriptie voltooid!")
            self.finished_signal.emit((segments, info, speaker_segments, transcript))
        
        except Exception as e:
            self.error.emit(f"Fout tijdens transcriptie: {str(e)}")


# Worker thread voor RSS feed verwerking
class RSSWorker(QThread):
    progress = pyqtSignal(str)
    feed_loaded = pyqtSignal(list)
    error = pyqtSignal(str)
    
    def __init__(self, feed_url):
        super().__init__()
        self.feed_url = feed_url
    
    def run(self):
        try:
            self.progress.emit(f"RSS feed laden: {self.feed_url}...")
            feed = feedparser.parse(self.feed_url)
            
            if not feed.entries:
                self.error.emit("Geen items gevonden in de RSS feed.")
                return
                
            episodes = []
            for entry in feed.entries:
                for link in entry.links:
                    if link.get('type', '').startswith('audio/'):
                        episodes.append({
                            'title': entry.title,
                            'url': link.href,
                            'published': entry.get('published', 'Onbekende datum'),
                            'description': entry.get('summary', 'Geen beschrijving')
                        })
                        break
            
            if not episodes:
                self.error.emit("Geen audio-items gevonden in de RSS feed.")
                return
                
            self.progress.emit(f"{len(episodes)} afleveringen gevonden.")
            self.feed_loaded.emit(episodes)
            
        except Exception as e:
            self.error.emit(f"Fout bij laden van RSS feed: {str(e)}")


# Hoofdvenster van de applicatie
class WHYcastTranscribeGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"WHYcast Transcribe v{VERSION}")
        self.resize(1000, 800)
        
        # Centrale widget en layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Tabs aanmaken
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)
        
        # Tab voor lokale bestanden
        self.local_tab = QWidget()
        self.tabs.addTab(self.local_tab, "Lokaal bestand")
        self.setup_local_tab()
        
        # Tab voor RSS feeds
        self.rss_tab = QWidget()
        self.tabs.addTab(self.rss_tab, "RSS Feed")
        self.setup_rss_tab()
        
        # Logger initialiseren
        self.setup_logger()
        
        # Status weergeven
        self.statusBar().showMessage("Gereed")
        
    def setup_local_tab(self):
        layout = QVBoxLayout(self.local_tab)
        
        # Bestandsselectie
        file_layout = QHBoxLayout()
        self.file_path = QLineEdit()
        self.file_path.setReadOnly(True)
        self.file_path.setPlaceholderText("Selecteer een audio bestand...")
        file_layout.addWidget(self.file_path)
        
        browse_button = QPushButton("Bladeren...")
        browse_button.clicked.connect(self.browse_file)
        file_layout.addWidget(browse_button)
        layout.addLayout(file_layout)
        
        # Transcriptie opties
        options_layout = QHBoxLayout()
        
        # Model selectie
        options_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["tiny", "base", "small", "medium", "large-v2", "large-v3"])
        self.model_combo.setCurrentText(MODEL_SIZE)
        options_layout.addWidget(self.model_combo)
        
        # Diarisatie opties
        self.diarize_checkbox = QCheckBox("Speaker diarisatie")
        self.diarize_checkbox.setChecked(USE_SPEAKER_DIARIZATION)
        options_layout.addWidget(self.diarize_checkbox)
        
        options_layout.addWidget(QLabel("Min sprekers:"))
        self.min_speakers_spin = QSpinBox()
        self.min_speakers_spin.setMinimum(1)
        self.min_speakers_spin.setMaximum(10)
        self.min_speakers_spin.setValue(DIARIZATION_MIN_SPEAKERS)
        options_layout.addWidget(self.min_speakers_spin)
        
        options_layout.addWidget(QLabel("Max sprekers:"))
        self.max_speakers_spin = QSpinBox()
        self.max_speakers_spin.setMinimum(1)
        self.max_speakers_spin.setMaximum(10)
        self.max_speakers_spin.setValue(DIARIZATION_MAX_SPEAKERS)
        options_layout.addWidget(self.max_speakers_spin)
        
        layout.addLayout(options_layout)
        
        # Start knop
        self.start_button = QPushButton("Start Transcriptie")
        self.start_button.clicked.connect(self.start_transcription)
        self.start_button.setEnabled(False)
        layout.addWidget(self.start_button)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Gereed")
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        
        # Splitter voor uitvoer en logging
        splitter = QSplitter(Qt.Vertical)
        
        # Transcriptie uitvoer
        self.output_group = QWidget()
        output_layout = QVBoxLayout(self.output_group)
        output_layout.addWidget(QLabel("<b>Transcriptie Uitvoer:</b>"))
        
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setFont(QFont("Consolas", 10))
        output_layout.addWidget(self.output_text)
        
        # Log uitvoer
        self.log_group = QWidget()
        log_layout = QVBoxLayout(self.log_group)
        log_layout.addWidget(QLabel("<b>Log Berichten:</b>"))
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
        log_layout.addWidget(self.log_text)
        
        # Widgets toevoegen aan splitter
        splitter.addWidget(self.output_group)
        splitter.addWidget(self.log_group)
        splitter.setSizes([500, 300])
        
        layout.addWidget(splitter)
        
    def setup_rss_tab(self):
        layout = QVBoxLayout(self.rss_tab)
        
        # Feed URL
        url_layout = QHBoxLayout()
        self.feed_url = QLineEdit()
        self.feed_url.setPlaceholderText("Voer RSS feed URL in...")
        # Set default RSS feed URI for 'why'
        self.feed_url.setText("https://whycast.podcast.audio/@whycast/feed.xml")
        url_layout.addWidget(self.feed_url)
        
        load_button = QPushButton("Laden")
        load_button.clicked.connect(self.load_rss_feed)
        url_layout.addWidget(load_button)
        layout.addLayout(url_layout)
        
        # Episode selectie
        layout.addWidget(QLabel("<b>Selecteer aflevering:</b>"))
        self.episodes_list = QListWidget()
        self.episodes_list.setSelectionMode(QListWidget.SingleSelection)
        layout.addWidget(self.episodes_list)
        
        # Download knop
        self.download_button = QPushButton("Download & Transcribeer Geselecteerde Aflevering")
        self.download_button.clicked.connect(self.download_and_transcribe)
        self.download_button.setEnabled(False)
        layout.addWidget(self.download_button)
        
        # Enable double-click to download and transcribe
        self.episodes_list.doubleClicked.connect(lambda _: self.download_and_transcribe())
        
        # Progress bar
        self.rss_progress_bar = QProgressBar()
        self.rss_progress_bar.setTextVisible(True)
        self.rss_progress_bar.setFormat("Gereed")
        self.rss_progress_bar.setValue(0)
        layout.addWidget(self.rss_progress_bar)
    
    def setup_logger(self):
        # Logging naar GUI text widget
        handler = QTextEditLogger(self.log_text)
        handler.setLevel(logging.INFO)
        
        # Root logger configureren
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.INFO)
    
    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Audio bestand selecteren", "",
            "Audio Bestanden (*.mp3 *.wav *.m4a *.ogg);;Alle bestanden (*.*)"
        )
        
        if file_path:
            self.file_path.setText(file_path)
            self.start_button.setEnabled(True)
    
    def start_transcription(self):
        audio_file = self.file_path.text()
        if not audio_file or not os.path.exists(audio_file):
            QMessageBox.warning(self, "Waarschuwing", "Selecteer eerst een geldig audio bestand.")
            return
        
        # UI elementen updaten
        self.start_button.setEnabled(False)
        self.progress_bar.setValue(10)
        self.progress_bar.setFormat("Verwerking...")
        self.output_text.clear()
        
        # Worker thread starten
        self.worker = TranscribeWorker(
            audio_file,
            self.model_combo.currentText(),
            self.diarize_checkbox.isChecked(),
            self.min_speakers_spin.value(),
            self.max_speakers_spin.value()
        )
        self.worker.progress.connect(self.update_progress)
        self.worker.finished_signal.connect(self.transcription_finished)
        self.worker.error.connect(self.show_error)
        self.worker.start()
    
    def load_rss_feed(self):
        feed_url = self.feed_url.text().strip()
        if not feed_url:
            QMessageBox.warning(self, "Waarschuwing", "Voer een geldige RSS feed URL in.")
            return
        
        # UI updaten
        self.episodes_list.clear()
        self.episodes_list.addItem("RSS feed wordt geladen...")
        self.rss_progress_bar.setValue(10)
        self.rss_progress_bar.setFormat("Laden...")
        
        # Worker thread starten
        self.rss_worker = RSSWorker(feed_url)
        self.rss_worker.progress.connect(self.update_rss_progress)
        self.rss_worker.feed_loaded.connect(self.display_episodes)
        self.rss_worker.error.connect(self.show_error)
        self.rss_worker.start()
    
    def display_episodes(self, episodes):
        self.episodes_list.clear()
        self.episodes = episodes
        for i, episode in enumerate(episodes):
            item = QListWidgetItem(f"[{i+1}] {episode['title']} ({episode['published']})")
            item.setData(Qt.UserRole, episode)
            # Render tooltip as HTML
            tooltip_html = f"""
            <b>Titel:</b> {episode['title']}<br>
            <b>Gepubliceerd:</b> {episode['published']}<br>
            <b>URL:</b> <a href='{episode['url']}'>{episode['url']}</a><br>
            <b>Beschrijving:</b> {episode['description']}
            """
            item.setToolTip(tooltip_html)
            self.episodes_list.addItem(item)
        self.download_button.setEnabled(True)
        self.rss_progress_bar.setValue(100)
        self.rss_progress_bar.setFormat(f"{len(episodes)} afleveringen gevonden")
    
    def download_and_transcribe(self):
        try:
            # Selecteer aflevering of kies er willekeurig een
            if not hasattr(self, 'episodes') or not self.episodes:
                return
            items = self.episodes_list.selectedItems()
            if items:
                episode = items[0].data(Qt.UserRole)
            else:
                episode = random.choice(self.episodes)
            # Download
            url = episode['url']
            file_name = os.path.basename(url)
            podcasts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', 'podcasts')
            os.makedirs(podcasts_dir, exist_ok=True)
            download_path = os.path.join(podcasts_dir, file_name)
            response = requests.get(url, stream=True)
            total_length = response.headers.get('content-length')
            if total_length is not None:
                total_length = int(total_length)
            downloaded = 0
            chunk_size = 8192
            self.rss_progress_bar.setValue(0)
            self.rss_progress_bar.setFormat("Downloaden...")
            with open(download_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        if total_length:
                            downloaded += len(chunk)
                            percent = int(downloaded * 100 / total_length)
                            self.rss_progress_bar.setValue(percent)
                            self.statusBar().showMessage(f"Download voortgang: {percent}%")
            self.statusBar().showMessage(f"Gedownload naar {download_path}")
            self.rss_progress_bar.setValue(100)
            self.rss_progress_bar.setFormat("Download voltooid")
            # Start transcriptie met gedownload bestand
            self.start_button.setEnabled(False)
            self.progress_bar.setValue(10)
            self.progress_bar.setFormat("Verwerking...")
            self.output_text.clear()
            self.worker = TranscribeWorker(
                download_path,
                self.model_combo.currentText(),
                self.diarize_checkbox.isChecked(),
                self.min_speakers_spin.value(),
                self.max_speakers_spin.value()
            )
            self.worker.progress.connect(self.update_progress)
            self.worker.finished_signal.connect(self.transcription_finished)
            self.worker.error.connect(self.show_error)
            self.worker.start()
        except Exception as e:
            tb_str = traceback.format_exc()
            self.show_error(f"Fout tijdens downloaden of starten van transcriptie: {str(e)}\n\nTraceback:\n{tb_str}")
    
    def update_progress(self, message):
        self.output_text.append(message)
        # Avoid moving the QTextCursor, which causes the QObject::connect warning
        # cursor = self.output_text.textCursor()
        # cursor.movePosition(QTextCursor.End)
        # self.output_text.setTextCursor(cursor)
    
    def update_rss_progress(self, message):
        self.statusBar().showMessage(message)
    
    def transcription_finished(self, results):
        segments, info, speaker_segments, transcript = results
        
        # Toon transcriptie
        self.output_text.append("\n--- Volledige Transcriptie ---\n")
        self.output_text.append(transcript)
        
        # Update UI
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat("Voltooid")
        self.start_button.setEnabled(True)
        
        QMessageBox.information(self, "Voltooid", 
                               f"Transcriptie is voltooid!\n\n"
                               f"Taal gedetecteerd: {info.language}\n"
                               f"Waarschijnlijkheid: {info.language_probability:.2f}\n"
                               f"Aantal segmenten: {len(segments)}")
    
    def show_error(self, error_message):
        # Log error to console and file
        print(f"[ERROR] {error_message}", file=sys.stderr)
        logging.error(error_message)
        # Also log the full traceback if available
        exc_type, exc_value, exc_tb = sys.exc_info()
        if exc_type is not None:
            tb_str = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
            print(tb_str, file=sys.stderr)
            logging.error(tb_str)
            error_message += f"\n\nTraceback:\n{tb_str}"
        QMessageBox.critical(self, "Fout", error_message)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Fout")
        self.start_button.setEnabled(True)

def main():
    # PyQt applicatie starten
    app = QApplication(sys.argv)
    window = WHYcastTranscribeGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()