import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyqtgraph as pg
import sounddevice as sd
import soundfile as sf
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class ConfigurationDialog(QDialog):
    """Dialog for configuring data folder and rater name on startup"""

    def __init__(self, parent=None, data_dir=None, rater_name=None):
        super().__init__(parent)
        self.setWindowTitle("Configuration")
        self.setModal(True)
        self.setFixedSize(500, 200)

        layout = QVBoxLayout(self)

        # Form layout for inputs
        form_layout = QFormLayout()

        # Data folder selection
        self.data_dir_edit = QLineEdit()
        self.data_dir_edit.setText(str(data_dir) if data_dir else "")
        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self.browse_data_folder)

        data_folder_layout = QHBoxLayout()
        data_folder_layout.addWidget(self.data_dir_edit)
        data_folder_layout.addWidget(self.browse_button)

        form_layout.addRow("Data Folder:", data_folder_layout)

        # Rater name input
        self.rater_name_edit = QLineEdit()
        self.rater_name_edit.setText(rater_name or "")
        form_layout.addRow("Rater Initials:", self.rater_name_edit)

        layout.addLayout(form_layout)

        # Buttons
        button_layout = QHBoxLayout()
        self.ok_button = QPushButton("OK")
        self.cancel_button = QPushButton("Cancel")

        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)

        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)

        # Set OK as default button
        self.ok_button.setDefault(True)

    def browse_data_folder(self):
        """Open folder selection dialog"""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Data Folder", self.data_dir_edit.text() or str(Path.home())
        )
        if folder:
            self.data_dir_edit.setText(folder)

    def get_data_dir(self):
        """Get the selected data directory"""
        return Path(self.data_dir_edit.text())

    def get_rater_name(self):
        """Get the entered rater name"""
        return self.rater_name_edit.text().strip()


class AudioAnnotator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.data_dir = None
        self.rater_name = None

        # Settings file path
        self.settings_file = Path("audiotag_settings.json")

        self.padding_s = 1.2
        self.loaded_padding_s = 6
        self.context_words = 4

        self.needed_columns = [
            "transcription",
            "word_clean",
            "pre_post",
            "start",
            "end",
            "rater",
            "changed",
        ]

        # audio init
        self.c_audio_data = None  # c for current
        self.c_sample_rate = None
        self.c_word_index = 0
        self.c_rating_df = None
        self.c_start_time = None
        self.c_end_time = None
        self.c_changed_times = False
        self.c_sub_id = None
        self.playback_line = None
        self.playback_timer = None

        self.init_ui()

        # Load settings or show configuration dialog
        if not self.load_settings():
            self.show_configuration_dialog()

        assert self.data_dir is not None, "Data directory not set"

        self.output_dir = self.data_dir / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_settings(self):
        """Load settings from JSON file. Returns True if successful, False otherwise."""
        try:
            if self.settings_file.exists():
                with open(self.settings_file, "r") as f:
                    settings = json.load(f)

                self.data_dir = Path(settings.get("data_dir", "data"))
                self.rater_name = settings.get("rater_name", "GKP")

                # Validate that the data directory exists
                if not self.data_dir.exists():
                    print(f"Warning: Data directory {self.data_dir} does not exist")
                    return False

                # Update sub list
                self.update_sub_list()
                self.update_current_rater()

                return True
        except Exception as e:
            print(f"Error loading settings: {e}")
        return False

    def save_settings(self):
        """Save current settings to JSON file."""
        try:
            settings = {"data_dir": str(self.data_dir), "rater_name": self.rater_name}
            with open(self.settings_file, "w") as f:
                json.dump(settings, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving settings: {e}")
            return False

    def show_configuration_dialog(self):
        """Show configuration dialog and update settings based on user input."""
        dialog = ConfigurationDialog(
            self, data_dir=self.data_dir, rater_name=self.rater_name
        )

        if dialog.exec() == QDialog.DialogCode.Accepted:
            # Get values from dialog
            new_data_dir = dialog.get_data_dir()
            new_rater_name = dialog.get_rater_name()

            # Validate inputs
            if not new_data_dir.exists():
                QMessageBox.warning(
                    self,
                    "Invalid Data Directory",
                    f"The selected directory does not exist: {new_data_dir}",
                )
                # Try again
                self.show_configuration_dialog()
                return

            if not new_rater_name:
                QMessageBox.warning(
                    self, "Invalid Rater Initials", "Please enter a rater initials."
                )
                # Try again
                self.show_configuration_dialog()
                return

            # Update settings
            self.data_dir = new_data_dir
            self.rater_name = new_rater_name

            # Update sub list
            self.update_sub_list()
            self.update_current_rater()

            # Save settings
            if not self.save_settings():
                QMessageBox.warning(
                    self,
                    "Settings Save Failed",
                    "Could not save settings. They will be lost when the "
                    "application closes.",
                )
        else:
            # User cancelled, exit application
            sys.exit(0)

    def init_ui(self):
        self.setWindowTitle("Audiotag")
        self.setGeometry(100, 100, 1600, 900)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)

        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        self.left_panel, self.left_panel_text_field = self.create_left_panel_file_list()
        self.right_panel_splitter = self.create_right_panel_rating()
        self.right_panel_splitter.setSizes([900, 300])

        splitter.addWidget(self.left_panel)
        splitter.addWidget(self.right_panel_splitter)

    def create_left_panel_file_list(self) -> tuple[QWidget, QLabel]:
        """Function that creates the left panel for the file list."""
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMaximumWidth(210)

        # Add loading button
        load_button = QPushButton("LOAD")
        load_button.clicked.connect(self.load_audio_file)
        configure_button = QPushButton("CONFIGURE")
        configure_button.clicked.connect(self.show_configuration_dialog)
        left_layout.addWidget(configure_button)
        left_layout.addWidget(load_button)

        # Add little text panel
        text_field = QLabel("Select a file to load")
        text_field.setWordWrap(True)
        text_field.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(text_field)

        # Add list
        self.sub_list = QListWidget()
        left_layout.addWidget(self.sub_list)

        return left_panel, text_field

    def create_right_panel_rating(self) -> QSplitter:
        """Function that creates the right panel for the rating of a sub"""
        right_panel_splitter = QSplitter(Qt.Orientation.Vertical)

        # top panel
        top_panel = QWidget()
        top_layout = QVBoxLayout(top_panel)

        self.plot_widget = pg.PlotWidget()
        top_layout.addWidget(self.plot_widget)
        self.plot_widget.setLabel("left", "Amplitude")
        self.plot_widget.setLabel("bottom", "Time (seconds)")

        right_panel_splitter.addWidget(top_panel)

        # bottom panel with text and buttons side by side
        bottom_panel = QWidget()
        bottom_layout = QHBoxLayout(bottom_panel)

        # Create context words panel (left of text edit)
        context_panel = QWidget()
        context_layout = QVBoxLayout(context_panel)
        context_panel.setMaximumWidth(200)
        context_panel.setMinimumWidth(150)
        context_panel.setMinimumHeight(200)  # Set minimum height

        # Context words list
        self.context_list = QListWidget()
        self.context_list.itemClicked.connect(self.on_context_item_clicked)
        context_layout.addWidget(self.context_list)

        # Add text panel
        self.word_text_edit = QTextEdit()
        self.word_text_edit.setPlaceholderText("")
        self.word_text_edit.setMaximumWidth(500)
        self.word_text_edit.setMinimumWidth(210)
        self.word_text_edit.setMinimumHeight(200)  # Set same minimum height
        font = self.word_text_edit.font()
        font.setPointSize(36)
        self.word_text_edit.setFont(font)

        # Create buttons & text fields
        self.play_button = QPushButton("PLAY WORD")
        self.play_context_button = QPushButton("PLAY CONTEXT")
        self.reset_button = QPushButton("RESET WORD")
        self.next_button = QPushButton("NEXT WORD")
        self.prev_button = QPushButton("PREV WORD")
        self.split_button = QPushButton("SPLIT WORD")
        self.delete_button = QPushButton("DELETE WORD")
        self.text_field_old_rater = QLabel("Last rater: N/A")
        font = self.text_field_old_rater.font()
        font.setPointSize(18)
        self.text_field_old_rater.setFont(font)
        self.text_field_old_rater.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.text_field_current_rater = QLabel(f"Current rater: {self.rater_name}")
        font = self.text_field_current_rater.font()
        font.setPointSize(18)
        self.text_field_current_rater.setFont(font)
        self.text_field_current_rater.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # disable buttons
        self.play_button.setEnabled(False)
        self.play_context_button.setEnabled(False)
        self.reset_button.setEnabled(False)
        self.next_button.setEnabled(False)
        self.prev_button.setEnabled(False)
        self.split_button.setEnabled(False)
        self.delete_button.setEnabled(False)

        # Set minimum height for all buttons to make them thicker
        def set_widget_vals(
            widgets: list[QWidget], height_min: int, width_max: int, width_min: int
        ):
            for widget in widgets:
                widget.setMinimumHeight(height_min)
                widget.setMaximumWidth(width_max)
                widget.setMinimumWidth(width_min)

        set_widget_vals(
            widgets=[
                self.next_button,
                self.prev_button,
                self.reset_button,
                self.split_button,
                self.delete_button,
                self.text_field_old_rater,
                self.text_field_current_rater,
            ],
            height_min=36,
            width_max=210,
            width_min=80,
        )
        set_widget_vals(
            widgets=[
                self.play_button,
                self.play_context_button,
            ],
            height_min=50,
            width_max=210,
            width_min=80,
        )

        # bind buttons to functions
        self.play_button.clicked.connect(self.play_audio)
        self.play_context_button.clicked.connect(self.play_context)
        self.reset_button.clicked.connect(self.reset_word)
        self.prev_button.clicked.connect(self.prev_word)
        self.next_button.clicked.connect(self.next_word)
        self.split_button.clicked.connect(self.split_word)
        self.delete_button.clicked.connect(self.delete_word)

        # Create two vertical button groups
        # Left column: Play and Reset buttons
        play_button_panel = QWidget()
        play_button_layout = QVBoxLayout(play_button_panel)
        play_button_layout.addStretch()
        play_button_layout.addWidget(self.text_field_old_rater)
        play_button_layout.addWidget(self.play_button)
        play_button_layout.addWidget(self.play_context_button)
        play_button_layout.addWidget(self.text_field_current_rater)
        play_button_layout.addStretch()

        # Middle column: Navigation and editing buttons
        navigation_button_panel = QWidget()
        navigation_button_layout = QVBoxLayout(navigation_button_panel)
        navigation_button_layout.addWidget(self.next_button)
        navigation_button_layout.addWidget(self.prev_button)
        navigation_button_layout.addWidget(self.reset_button)
        navigation_button_layout.addWidget(self.split_button)
        navigation_button_layout.addWidget(self.delete_button)
        navigation_button_layout.addStretch()

        # Add both button panels to main layout
        bottom_layout.addWidget(context_panel)
        bottom_layout.addWidget(navigation_button_panel)
        bottom_layout.addWidget(play_button_panel)
        bottom_layout.addWidget(self.word_text_edit)

        right_panel_splitter.addWidget(bottom_panel)

        return right_panel_splitter

    def update_current_rater(self):
        """Updates the current rater text field on the right"""
        self.text_field_current_rater.setText(f"Current rater: {self.rater_name}")

    def update_sub_list(self):
        """Updates the sub list on the left"""
        assert self.data_dir is not None, "Data directory not set"

        # Add some sample items
        self.sub_list.clear()
        rating_files = sorted(list((self.data_dir / "ratings").glob("*.csv")))
        for rating_file in rating_files:
            sub_id = "-".join(rating_file.stem.split("-")[0:2])
            self.sub_list.addItem(QListWidgetItem(sub_id))

    def show_current_word(self):
        """Function that sets current word audio segment and plots it."""

        if (
            self.c_rating_df is None
            or self.c_audio_data is None
            or self.c_sample_rate is None
        ):
            QMessageBox.warning(
                self,
                "Data not loaded",
                f"Data not loaded:\n{self.c_rating_df=}"
                f"\n{self.c_audio_data=}\n{self.c_sample_rate=}",
            )
            return

        # set word text
        self.word_text_edit.setText(
            self.c_rating_df.loc[self.c_word_index, "transcription"]
        )

        # update word counter & rater text field
        last_rater = self.c_rating_df.loc[self.c_word_index, "rater"]
        last_rater = last_rater or "N/A"
        self.text_field_old_rater.setText(f"Last rater: {last_rater}")

        # get indices
        self.c_start_time = self.c_rating_df.loc[self.c_word_index, "start"]
        self.c_end_time = self.c_rating_df.loc[self.c_word_index, "end"]
        start_time_loaded_padding = max(0, self.c_start_time - self.loaded_padding_s)
        end_time_loaded_padding = min(
            len(self.c_audio_data), self.c_end_time + self.loaded_padding_s
        )
        idx_start_padded = int(start_time_loaded_padding * self.c_sample_rate)
        idx_end_padded = int(end_time_loaded_padding * self.c_sample_rate)

        # plot padded audio
        y_audio_data = self.c_audio_data[idx_start_padded:idx_end_padded]
        x_time_axis = np.linspace(
            start_time_loaded_padding, end_time_loaded_padding, len(y_audio_data)
        )
        self.plot_widget.clear()
        self.plot_widget.plot(x_time_axis, y_audio_data, pen="w")

        # set the range of the viewport
        start_time_padded = max(0, self.c_start_time - self.padding_s)
        end_time_padded = min(len(self.c_audio_data), self.c_end_time + self.padding_s)
        self.plot_widget.setXRange(start_time_padded, end_time_padded)

        # Lock y-axis to prevent scrolling
        self.plot_widget.setMouseEnabled(x=True, y=False)
        self.plot_widget.setYRange(self.c_audio_y_min, self.c_audio_y_max)

        # plot start and end lines (draggable)
        self.start_line = pg.InfiniteLine(
            pos=self.c_start_time,
            pen=pg.mkPen("g", width=3),
            hoverPen=pg.mkPen("orange", width=3),
            movable=True,
        )
        self.end_line = pg.InfiniteLine(
            pos=self.c_end_time,
            pen=pg.mkPen("r", width=3),
            hoverPen=pg.mkPen("orange", width=3),
            movable=True,
        )

        # Connect line movement to update times
        self.start_line.sigPositionChanged.connect(self.on_start_line_moved)
        self.end_line.sigPositionChanged.connect(self.on_end_line_moved)

        self.plot_widget.addItem(self.start_line)
        self.plot_widget.addItem(self.end_line)

        # show list of previous and next words
        self.context_list.clear()
        context_words = self.c_rating_df.loc[:, "transcription"].tolist()
        context_indices = list(range(len(self.c_rating_df)))
        current_item = None
        for idx_word, word in zip(context_indices, context_words):
            if idx_word == self.c_word_index:
                item = QListWidgetItem(f"{idx_word:03d} | ▶ {word}")
                item.setBackground(Qt.GlobalColor.darkGreen)
                item.setForeground(Qt.GlobalColor.white)
                current_item = item
            else:
                item = QListWidgetItem(f"{idx_word:03d} |  {word}")
            self.context_list.addItem(item)

        # Scroll to the current word
        if current_item is not None:
            self.context_list.scrollToItem(
                current_item, QListWidget.ScrollHint.PositionAtCenter
            )

        # enable/disable buttons
        self.next_button.setEnabled(True)
        if self.c_word_index == len(self.c_rating_df) - 1:
            self.next_button.setText("SAVE")
        else:
            self.next_button.setText("NEXT WORD")
        self.prev_button.setEnabled(self.c_word_index > 0)
        self.split_button.setEnabled(True)
        self.delete_button.setEnabled(len(self.c_rating_df) > 1)
        self.play_button.setEnabled(True)
        self.play_context_button.setEnabled(True)
        self.reset_button.setEnabled(True)

    def on_context_item_clicked(self, item):
        """Handle click on context list item to jump to that word"""
        if self.c_rating_df is None:
            return

        self.save_rating()

        # Extract word index from the item text (format: "000 | word")
        item_text = item.text()
        try:
            word_index = int(item_text.split(" | ")[0])
            if 0 <= word_index < len(self.c_rating_df):
                self.c_word_index = word_index
                self.show_current_word()
        except (ValueError, IndexError):
            # If parsing fails, ignore the click
            QMessageBox.warning(
                self,
                "Navigation failed, use next/prev buttons",
                "Navigation failed, use next/prev buttons",
            )
            pass

    def on_start_line_moved(self, line):
        """Called when the start line is dragged"""
        new_start_time = line.value()

        # Prevent start line from going beyond end line
        if self.c_end_time is not None and new_start_time >= self.c_end_time:
            # Reset to previous position
            line.setValue(self.c_start_time)
            return

        self.c_start_time = new_start_time
        self.c_changed_times = True

    def on_end_line_moved(self, line):
        """Called when the end line is dragged"""
        new_end_time = line.value()

        # Prevent end line from going before start line
        if self.c_start_time is not None and new_end_time <= self.c_start_time:
            # Reset to previous position
            line.setValue(self.c_end_time)
            return

        self.c_end_time = new_end_time
        self.c_changed_times = True

    def start_playback_tracking(self, start_time, end_time):
        """Start tracking playback position with a blue line"""
        # Remove existing playback line if any
        if self.playback_line is not None:
            self.plot_widget.removeItem(self.playback_line)

        # Create blue playback line
        self.playback_line = pg.InfiniteLine(
            pos=start_time, pen=pg.mkPen("b", width=2), movable=False
        )
        self.plot_widget.addItem(self.playback_line)

        # Create timer to update playback position
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(
            lambda: self.update_playback_position(start_time, end_time)
        )
        self.playback_timer.start(10)  # Update every 10ms

        # Store playback start time
        self.playback_start_time = start_time
        self.playback_current_time = start_time

    def update_playback_position(self, start_time, end_time):
        """Update the blue line position during playback"""
        if self.playback_line is None:
            return

        # Calculate elapsed time (approximate)
        self.playback_current_time += 0.01  # 10ms = 0.01 seconds

        # Update line position
        self.playback_line.setValue(self.playback_current_time)

        # Stop tracking when playback reaches end
        if self.playback_current_time >= end_time:
            self.stop_playback_tracking()

    def stop_playback_tracking(self):
        """Stop playback tracking and remove the blue line"""
        if self.playback_timer is not None:
            self.playback_timer.stop()
            self.playback_timer = None

        if self.playback_line is not None:
            self.plot_widget.removeItem(self.playback_line)
            self.playback_line = None

    def play_audio(self, with_padding: bool = False):
        """Play the audio from start to end of current word"""
        if self.c_start_time is None or self.c_end_time is None:
            QMessageBox.warning(
                self,
                "Start or end time not set",
                "Start or end time not set",
            )
            return
        if self.c_audio_data is None or self.c_sample_rate is None:
            QMessageBox.warning(
                self,
                "Audio data not loaded",
                "Audio data not loaded",
            )
            return

        print("PLAYING AUDIO")
        if with_padding:
            start_time = self.c_start_time - self.padding_s
            end_time = self.c_end_time + self.padding_s
        else:
            start_time = self.c_start_time
            end_time = self.c_end_time

        idx_start = int(start_time * self.c_sample_rate)
        idx_end = int(end_time * self.c_sample_rate)
        word_audio = self.c_audio_data[idx_start:idx_end]

        sd.play(word_audio, self.c_sample_rate)

        # Create and start playback line tracking
        self.start_playback_tracking(start_time, end_time)

    def play_context(self):
        """Play the audio with padding around current word"""
        self.play_audio(with_padding=True)

    def split_word(self):
        """Split the current word into two new words"""
        if (
            self.c_rating_df is None
            or self.c_start_time is None
            or self.c_end_time is None
        ):
            QMessageBox.warning(
                self,
                "No data loaded",
                "No data loaded",
            )
            return

        # split df into two
        first_df = self.c_rating_df.iloc[: self.c_word_index + 1]
        second_df = self.c_rating_df.iloc[self.c_word_index + 1 :]
        new_row_df = first_df.loc[[self.c_word_index]]
        # concat with new row
        self.c_rating_df = pd.concat([first_df, new_row_df, second_df]).reset_index(
            drop=True
        )
        # update start and end times
        self.c_rating_df.loc[self.c_word_index, "end"] = (
            self.c_start_time + self.c_end_time
        ) / 2 - 0.1
        self.c_rating_df.loc[self.c_word_index + 1, "start"] = (
            self.c_start_time + self.c_end_time
        ) / 2 + 0.1
        # need to save new df
        self.c_rating_df.to_csv(
            self.output_dir / f"{self.c_sub_id}-free_association_carver_rated.csv",
            index=False,
        )
        self.show_current_word()

    def delete_word(self):
        """Delete the current word"""
        if self.c_rating_df is None:
            QMessageBox.warning(
                self,
                "No data loaded",
                "No data loaded",
            )
            return
        # confirm deletion
        reply = QMessageBox.question(
            self,
            "Delete word",
            "Are you sure?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        # delete row
        self.c_rating_df = self.c_rating_df.drop(self.c_word_index).reset_index(
            drop=True
        )

        self.c_rating_df.to_csv(
            self.output_dir / f"{self.c_sub_id}-free_association_carver_rated.csv",
            index=False,
        )
        self.show_current_word()

    def normalize_word(self, word: str) -> str:
        return word.lower().strip()

    def save_rating(self):
        if (
            self.c_rating_df is None
            or self.c_audio_data is None
            or self.c_sample_rate is None
        ):
            QMessageBox.warning(
                self,
                "Data not loaded",
                f"Data not loaded:\n{self.c_rating_df=}"
                f"\n{self.c_audio_data=}\n{self.c_sample_rate=}",
            )
            return

        previous_word = self.normalize_word(
            self.c_rating_df.loc[self.c_word_index, "transcription"]
        )
        current_word = self.normalize_word(self.word_text_edit.toPlainText())

        # only if something changed, then save
        if previous_word != current_word or self.c_changed_times:
            # save
            self.c_rating_df.loc[self.c_word_index, "transcription"] = current_word
            self.c_rating_df.loc[self.c_word_index, "rater"] = self.rater_name
            self.c_rating_df.loc[self.c_word_index, "start"] = self.c_start_time
            self.c_rating_df.loc[self.c_word_index, "end"] = self.c_end_time
            self.c_changed_times = False
            self.c_rating_df.loc[self.c_word_index, "changed"] = True
        self.c_rating_df.to_csv(
            self.output_dir / f"{self.c_sub_id}-free_association_carver_rated.csv",
            index=False,
        )

    def next_word(self):
        """Function that is called when the NEXT WORD button is clicked."""

        if self.c_rating_df is None:
            QMessageBox.warning(
                self,
                "No data loaded",
                "No data loaded",
            )
            return

        self.save_rating()
        if self.c_word_index == len(self.c_rating_df) - 1:
            QMessageBox.information(
                self,
                "Data saved!",
                "Data saved!",
            )
            return

        self.c_word_index += 1
        self.show_current_word()

    def prev_word(self):
        """Function that is called when the PREVIOUS WORD button is clicked."""
        self.save_rating()
        if self.c_word_index == 0:
            QMessageBox.warning(
                self,
                "No previous word",
                "No previous word",
            )
            return
        self.c_word_index -= 1
        self.show_current_word()

    def reset_word(self):
        """When RESET WORD button is clicked, reload word rating and times"""
        self.c_changed_times = False
        self.show_current_word()

    def load_audio_file(self):
        """Function that is called when the LOAD button is clicked."""
        assert self.data_dir is not None, "Data directory not set"

        curr_item = self.sub_list.currentItem()
        if curr_item is not None:
            sub_id = curr_item.text()
            self.c_sub_id = sub_id
            self.left_panel_text_field.setText(f"Loading {sub_id}...")

            # if rating file already exists, load it
            rating_file = (
                self.output_dir / f"{sub_id}-free_association_carver_rated.csv"
            )
            if rating_file.exists():
                rating_df = pd.read_csv(rating_file)
                print(f"Loaded existing rating data: {rating_file}")
            else:
                # get previous rating file
                old_rating_file = (
                    self.data_dir / "ratings" / f"{sub_id}-free_association_carver.csv"
                )
                rating_df = pd.read_csv(old_rating_file)
                rating_df["changed"] = False
                rating_df.to_csv(rating_file, index=False)
                print("Loaded old ratings and saved as output")
            self.c_rating_df = rating_df

            for column in self.needed_columns:
                assert column in rating_df.columns, (
                    f"{column} column not found in rating data"
                )

            # get audio
            audio_file = self.data_dir / "audio" / f"{sub_id}_carver.wav"
            if not audio_file.exists():
                full_audio_file = audio_file.resolve()
                QMessageBox.warning(
                    self,
                    "Audio file not found",
                    f"Audio file not found: {full_audio_file}",
                )
                self.left_panel_text_field.setText(f"Audio not found for {sub_id}")
                return

            self.c_audio_data, self.c_sample_rate = sf.read(audio_file)

            self.c_audio_y_max = np.max(self.c_audio_data) * 1.1
            self.c_audio_y_min = np.min(self.c_audio_data) * 1.1

            self.left_panel_text_field.setText(f"Successfully loaded {sub_id}!")

            self.c_word_index = 0
            self.show_current_word()


def main():
    app = QApplication(sys.argv)

    # Set dark theme
    app.setStyle("Fusion")

    # Apply dark color palette
    from PyQt6.QtGui import QColor

    dark_palette = app.palette()

    # Define dark colors
    dark_gray = QColor(53, 53, 53)
    darker_gray = QColor(25, 25, 25)
    light_gray = QColor(180, 180, 180)
    blue = QColor(42, 130, 218)

    # Set dark colors for different UI elements
    dark_palette.setColor(dark_palette.ColorRole.Window, dark_gray)
    dark_palette.setColor(dark_palette.ColorRole.WindowText, light_gray)
    dark_palette.setColor(dark_palette.ColorRole.Base, darker_gray)
    dark_palette.setColor(dark_palette.ColorRole.AlternateBase, dark_gray)
    dark_palette.setColor(dark_palette.ColorRole.ToolTipBase, dark_gray)
    dark_palette.setColor(dark_palette.ColorRole.ToolTipText, light_gray)
    dark_palette.setColor(dark_palette.ColorRole.Text, light_gray)
    dark_palette.setColor(dark_palette.ColorRole.Button, dark_gray)
    dark_palette.setColor(dark_palette.ColorRole.ButtonText, light_gray)
    dark_palette.setColor(dark_palette.ColorRole.BrightText, light_gray)
    dark_palette.setColor(dark_palette.ColorRole.Link, blue)
    dark_palette.setColor(dark_palette.ColorRole.Highlight, blue)
    dark_palette.setColor(dark_palette.ColorRole.HighlightedText, light_gray)

    app.setPalette(dark_palette)

    window = AudioAnnotator()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
