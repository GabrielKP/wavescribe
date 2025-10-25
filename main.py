# main.py
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyqtgraph as pg
import sounddevice as sd
import soundfile as sf
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
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


class AudioAnnotator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.data_dir = Path("data")
        self.output_dir = self.data_dir / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rater_name = "GKP"
        self.padding_s = 2

        self.needed_columns = [
            "transcription",
            "word_clean",
            "pre_post",
            "start",
            "end",
            "rater",
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

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Audiotag")
        self.setGeometry(100, 100, 1600, 900)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)

        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        self.left_panel, self.file_list, self.left_panel_text_field = (
            self.create_left_panel_file_list()
        )
        self.right_panel_splitter = self.create_right_panel_rating()
        self.right_panel_splitter.setSizes([900, 300])

        splitter.addWidget(self.left_panel)
        splitter.addWidget(self.right_panel_splitter)

    def create_left_panel_file_list(self) -> tuple[QWidget, QListWidget, QLabel]:
        """Function that creates the left panel for the file list."""
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMaximumWidth(210)

        # Add loading button
        load_button = QPushButton("LOAD")
        load_button.clicked.connect(self.load_audio_file)
        left_layout.addWidget(load_button)

        # Add little text panel
        text_field = QLabel("Select a file to load")
        text_field.setWordWrap(True)
        text_field.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(text_field)

        # Add list
        file_list = QListWidget()
        left_layout.addWidget(file_list)

        # Add some sample items
        rating_files = sorted(list((self.data_dir / "ratings").glob("*.csv")))
        for rating_file in rating_files:
            sub_id = "-".join(rating_file.stem.split("-")[0:2])
            file_list.addItem(QListWidgetItem(sub_id))

        return left_panel, file_list, text_field

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

        # add text panel
        self.word_text_edit = QTextEdit()
        self.word_text_edit.setPlaceholderText("")
        self.word_text_edit.setMaximumWidth(500)
        font = self.word_text_edit.font()
        font.setPointSize(36)
        self.word_text_edit.setFont(font)
        bottom_layout.addWidget(self.word_text_edit)

        # Create buttons & text fields
        self.text_field_counter = QLabel("Select a file to load")
        font = self.text_field_counter.font()
        font.setPointSize(18)
        self.text_field_counter.setFont(font)
        self.text_field_counter.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.play_button = QPushButton("PLAY WORD")
        self.play_context_button = QPushButton("PLAY CONTEXT")
        self.reset_button = QPushButton("RESET")
        self.next_button = QPushButton("NEXT WORD")
        self.prev_button = QPushButton("PREV WORD")
        self.split_button = QPushButton("SPLIT WORD")
        self.delete_button = QPushButton("DELETE WORD")
        self.text_field_rater = QLabel("LAST RATER: N/A")
        font = self.text_field_rater.font()
        font.setPointSize(18)
        self.text_field_rater.setFont(font)
        self.text_field_rater.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Set minimum height for all buttons to make them thicker
        button_height = 36
        for qthing in [
            self.text_field_counter,
            self.play_button,
            self.reset_button,
            self.play_context_button,
            self.next_button,
            self.prev_button,
            self.split_button,
            self.delete_button,
            self.text_field_rater,
        ]:
            qthing.setMinimumHeight(button_height)
            qthing.setFixedWidth(210)

        # bind buttons to functions
        self.play_button.clicked.connect(self.play_audio)
        # self.play_context_button.clicked.connect(self.play_context)
        self.reset_button.clicked.connect(self.reset_word)
        self.prev_button.clicked.connect(self.prev_word)
        self.next_button.clicked.connect(self.next_word)
        # self.split_button.clicked.connect(self.split_word)
        # self.delete_button.clicked.connect(self.delete_word)

        # Create two vertical button groups
        # Left column: Play and Reset buttons
        left_button_panel = QWidget()
        left_button_layout = QVBoxLayout(left_button_panel)
        left_button_layout.addWidget(self.text_field_counter)
        left_button_layout.addWidget(self.play_button)
        left_button_layout.addWidget(self.play_context_button)
        left_button_layout.addWidget(self.reset_button)
        left_button_layout.addStretch()

        # Middle column: Navigation and editing buttons
        middle_button_panel = QWidget()
        middle_button_layout = QVBoxLayout(middle_button_panel)
        middle_button_layout.addWidget(self.next_button)
        middle_button_layout.addWidget(self.prev_button)
        middle_button_layout.addWidget(self.split_button)
        middle_button_layout.addWidget(self.delete_button)
        middle_button_layout.addStretch()

        # Right column: Rater text field
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.addWidget(self.text_field_rater)
        right_layout.addStretch()

        # Add both button panels to main layout
        bottom_layout.addWidget(left_button_panel)
        bottom_layout.addWidget(middle_button_panel)
        bottom_layout.addWidget(right_panel)
        bottom_layout.addStretch()

        right_panel_splitter.addWidget(bottom_panel)

        return right_panel_splitter

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
        self.text_field_counter.setText(
            f"{self.c_word_index + 1}/{len(self.c_rating_df)}"
        )
        self.text_field_rater.setText(f"Last rater: {last_rater}")

        # get indices
        self.c_start_time = self.c_rating_df.loc[self.c_word_index, "start"]
        self.c_end_time = self.c_rating_df.loc[self.c_word_index, "end"]
        start_time_padded = max(0, self.c_start_time - self.padding_s)
        end_time_padded = min(len(self.c_audio_data), self.c_end_time + self.padding_s)
        idx_start_padded = int(start_time_padded * self.c_sample_rate)
        idx_end_padded = int(end_time_padded * self.c_sample_rate)

        # plot padded audio
        y_audio_data = self.c_audio_data[idx_start_padded:idx_end_padded]
        x_time_axis = np.linspace(start_time_padded, end_time_padded, len(y_audio_data))
        self.plot_widget.clear()
        self.plot_widget.plot(x_time_axis, y_audio_data, pen="w")
        self.plot_widget.setXRange(start_time_padded, end_time_padded)

        # Lock y-axis to prevent scrolling
        self.plot_widget.setMouseEnabled(
            x=True, y=False
        )  # Disable y-axis mouse interaction

        # plot start and end lines
        self.plot_widget.addLine(x=self.c_start_time, pen=pg.mkPen("r", width=2))
        self.plot_widget.addLine(x=self.c_end_time, pen=pg.mkPen("g", width=2))

    def play_audio(self):
        """Function that is called when the PLAY AUDIO button is clicked."""
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
        idx_start = int(self.c_start_time * self.c_sample_rate)
        idx_end = int(self.c_end_time * self.c_sample_rate)
        word_audio = self.c_audio_data[idx_start:idx_end]
        sd.play(word_audio, self.c_sample_rate)

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
            self.c_changed_times = False
        self.c_rating_df.loc[self.c_word_index, "rated"] = True
        self.c_rating_df.to_csv(
            self.output_dir / f"{self.c_sub_id}-free_association_carver_rated.csv",
            index=False,
        )

    def next_word(self):
        """Function that is called when the NEXT WORD button is clicked."""

        self.save_rating()
        if (
            self.c_rating_df is not None
            and self.c_word_index == len(self.c_rating_df) - 1
        ):
            QMessageBox.warning(
                self,
                "No next word",
                "No next word",
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
        curr_item = self.file_list.currentItem()
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
                rating_df["rated"] = False
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

            self.left_panel_text_field.setText(f"Successfully loaded {sub_id}!")

            self.c_word_index = 0
            self.show_current_word()


def main():
    app = QApplication(sys.argv)

    # Set dark theme (optional, looks good for audio apps)
    app.setStyle("Fusion")

    window = AudioAnnotator()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
