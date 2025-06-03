import sys
import json
import threading
import time
from typing import Literal
import cv2
import numpy as np
import onnxruntime
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QButtonGroup,
    QPushButton,
    QMessageBox,
    QSpacerItem,
    QSizePolicy,
)
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import QScreen
from datetime import datetime
import os


class ModernMentalHealthSurveyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mental Health Quick Check")

        # --- Define Log Directory ---
        self.log_directory = "logs"
        try:
            os.makedirs(self.log_directory, exist_ok=True)  # Create if it doesn't exist
            print(
                f"Logs will be saved in directory: '{os.path.abspath(self.log_directory)}'"
            )
        except OSError as e:
            print(
                f"Error creating log directory '{self.log_directory}': {e}. Logs will be saved in current directory."
            )
            # Fallback to current directory if creation fails
            self.log_directory = "."

        # --- ONNX Model Configuration (USER ACTION REQUIRED) ---
        self.onnx_model_path = "model.onnx"
        self.class_labels = [
            "anger",
            "contempt",
            "disgust",
            "embarrass",
            "fear",
            "joy",
            "neutral",
            "pride",
            "sadness",
            "surprise",
        ]
        self.input_size = (224, 224)  # Expected input H, W for the model
        self.ort_session = None
        self.input_name = None
        self.output_name = None
        self._load_onnx_model()
        # --- End ONNX Configuration ---

        self.questions = [
            {
                "text": "1. Over the last 2 weeks, have you often been bothered by feeling down, depressed, or hopeless?",
                "options": ["Yes", "No"],
            },
            {
                "text": "2. Over the last 2 weeks, have you often been bothered by little interest or pleasure in doing things?",
                "options": ["Yes", "No"],
            },
            {
                "text": "3. Over the last 2 weeks, have you often been bothered by feeling nervous, anxious, or on edge?",
                "options": ["Yes", "No"],
            },
            {
                "text": "4. Over the last 2 weeks, have you often been bothered by not being able to stop or control worrying?",
                "options": ["Yes", "No"],
            },
            {
                "text": "5. Over the last 2 weeks, have you often been bothered by having trouble relaxing?",
                "options": ["Yes", "No"],
            },
        ]
        self.num_questions = len(self.questions)
        self.current_question_index = 0
        self.user_answers = [None] * self.num_questions

        # Generate unique log filenames for this session
        base_timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Adding milliseconds to prediction log for higher chance of uniqueness if app restarts very quickly
        # prediction_timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        self.survey_log_file_name = os.path.join(
            self.log_directory, f"survey_log_{base_timestamp_str}.json"
        )
        self.prediction_log_file_name = os.path.join(
            self.log_directory, f"prediction_log_{base_timestamp_str}.json"
        )

        print(f"Current session survey log file: {self.survey_log_file_name}")
        print(f"Current session prediction log file: {self.prediction_log_file_name}")

        self._log_event(
            action_type="passive", event_type="app_init"
        )  # Logs to survey_log_file_name

        self.capture_active = False
        self.capture_thread = None

        self.init_ui()
        self.apply_styles()
        self._center_window()

        self._start_webcam_capture()
        self.display_question()
        self._log_event(
            action_type="passive",
            event_type="question_displayed",
            details={"question_index": self.current_question_index + 1},
        )  # Logs to survey_log_file_name

    def _load_onnx_model(self):
        # ... (same as before) ...
        if not os.path.exists(self.onnx_model_path):
            print(
                f"ONNX Model Error: File not found at '{self.onnx_model_path}'. Inference will be disabled."
            )
            self.ort_session = None
            return
        try:
            self.ort_session = onnxruntime.InferenceSession(self.onnx_model_path)
            self.input_name = self.ort_session.get_inputs()[0].name
            self.output_name = self.ort_session.get_outputs()[0].name
            print(f"ONNX model '{self.onnx_model_path}' loaded successfully.")
            print(
                f"Model Input Name: {self.input_name}, Output Name: {self.output_name}"
            )
        except Exception as e:
            print(f"Error loading ONNX model: {e}")
            self.ort_session = None

    def _center_window(self):
        # ... (same as before) ...
        screen = QApplication.primaryScreen()
        if screen:
            screen_geometry = screen.availableGeometry()
            center_point = screen_geometry.center()
            self.move(
                center_point.x() - self.width() // 2,
                center_point.y() - self.height() // 2,
            )

    def _log_event(
        self, action_type: Literal["active", "passive"], event_type: str, details=None
    ):
        # This logs to self.survey_log_file_name
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_entry = {
            "timestamp": timestamp,
            "action_type": action_type,
            "event_type": event_type,
        }
        if details:
            log_entry["details"] = details

        all_logs = []
        try:
            if (
                os.path.exists(self.survey_log_file_name)
                and os.path.getsize(self.survey_log_file_name) > 0
            ):
                with open(self.survey_log_file_name, "r", encoding="utf-8") as f:
                    all_logs = json.load(f)
                if not isinstance(all_logs, list):
                    all_logs = []
            else:
                all_logs = []
        except json.JSONDecodeError:
            all_logs = []
        except Exception:
            all_logs = []

        all_logs.append(log_entry)
        try:
            with open(self.survey_log_file_name, "w", encoding="utf-8") as f:
                json.dump(all_logs, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(
                f"Error writing to survey log file '{self.survey_log_file_name}': {e}"
            )

    def _log_image_prediction(
        self, predicted_label: str, confidence: float, predicted_index: int
    ):
        # This new method logs to self.prediction_log_file_name
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_entry = {
            "timestamp": timestamp,
            "predicted_label": predicted_label,
            "confidence": round(confidence, 4),
            "predicted_index": int(predicted_index),
            # You could add more details here if needed, e.g., raw scores
        }

        all_predictions = []
        try:
            if (
                os.path.exists(self.prediction_log_file_name)
                and os.path.getsize(self.prediction_log_file_name) > 0
            ):
                with open(self.prediction_log_file_name, "r", encoding="utf-8") as f:
                    all_predictions = json.load(f)
                if not isinstance(all_predictions, list):
                    print(
                        f"Warning: Prediction log file '{self.prediction_log_file_name}' was not a list. Resetting."
                    )
                    all_predictions = []
            else:  # File doesn't exist yet for this session or is empty
                all_predictions = []
        except json.JSONDecodeError:
            print(
                f"Warning: Could not decode JSON from '{self.prediction_log_file_name}'. Starting with a new prediction log list."
            )
            all_predictions = []
        except Exception as e:
            print(
                f"Error reading prediction log file '{self.prediction_log_file_name}': {e}. Starting with a new list."
            )
            all_predictions = []

        all_predictions.append(log_entry)

        try:
            with open(self.prediction_log_file_name, "w", encoding="utf-8") as f:
                json.dump(all_predictions, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(
                f"Error writing to prediction log file '{self.prediction_log_file_name}': {e}"
            )

    def _preprocess_image(self, frame: np.ndarray) -> np.ndarray:
        # ... (same as before, with mean/std normalization) ...
        img = cv2.resize(frame, self.input_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img

    def _start_webcam_capture(self):
        # ... (same as before) ...
        if self.capture_thread is None:
            self.capture_active = True
            self.capture_thread = threading.Thread(
                target=self._webcam_capture_loop, daemon=True
            )
            self.capture_thread.start()
            print("Webcam capture thread started.")

    def _webcam_capture_loop(self):
        # ... (modified to call _log_image_prediction) ...
        cap = None
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print(f"{datetime.now()}: Error: Could not open webcam.")
                self.capture_active = False
                return
            print(f"{datetime.now()}: Webcam opened successfully.")

            while self.capture_active:
                ret, frame = cap.read()
                if ret:
                    if self.ort_session:
                        try:
                            preprocessed_frame = self._preprocess_image(frame.copy())
                            ort_inputs = {self.input_name: preprocessed_frame}
                            ort_outs = self.ort_session.run(
                                [self.output_name], ort_inputs
                            )
                            scores = ort_outs[0][0]
                            predicted_index = np.argmax(scores)
                            confidence = float(scores[predicted_index])
                            predicted_label = "Unknown"
                            if 0 <= predicted_index < len(self.class_labels):
                                predicted_label = self.class_labels[predicted_index]

                            # print(f"{datetime.now()}: Predicted Label: {predicted_label} (Confidence: {confidence:.2f})") # Optional console print

                            # Log to the dedicated prediction log file
                            self._log_image_prediction(
                                predicted_label, confidence, predicted_index
                            )

                        except Exception as e:
                            print(
                                f"{datetime.now()}: Error during model inference: {e}"
                            )
                else:
                    print(
                        f"{datetime.now()}: Error: Failed to capture frame from webcam."
                    )

                for _ in range(10):
                    if not self.capture_active:
                        break
                    time.sleep(0.1)
        except Exception as e:
            print(f"{datetime.now()}: Exception in webcam loop: {e}")
        finally:
            if cap and cap.isOpened():
                cap.release()
            print(
                f"{datetime.now()}: Webcam capture thread finished and webcam released."
            )
            self.capture_active = False

    def _stop_webcam_capture(self):
        # ... (same as before) ...
        print("Attempting to stop webcam capture thread...")
        self.capture_active = False
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.5)
            if self.capture_thread.is_alive():
                print("Webcam capture thread did not stop in time.")
            else:
                print("Webcam capture thread stopped.")
        self.capture_thread = None

    def init_ui(self):
        # ... (UI setup - same as before) ...
        self.setObjectName("MainWindow")
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(20, 20, 20, 20)
        self.main_layout.setSpacing(20)
        self.disclaimer_label = QLabel(
            "<b>Important:</b> This is NOT a diagnostic tool. Consult a professional for any mental health concerns."
        )
        self.disclaimer_label.setObjectName("DisclaimerLabel")
        self.disclaimer_label.setWordWrap(True)
        self.main_layout.addWidget(self.disclaimer_label)
        self.instructions_label = QLabel(
            "Please answer the following questions based on how you have felt over the <b>last 2 weeks</b>."
        )
        self.instructions_label.setObjectName("InstructionsLabel")
        self.instructions_label.setWordWrap(True)
        self.main_layout.addWidget(self.instructions_label)
        self.question_area_widget = QWidget()
        self.question_area_widget.setObjectName("QuestionAreaWidget")
        question_area_layout = QVBoxLayout(self.question_area_widget)
        question_area_layout.setContentsMargins(20, 15, 20, 15)
        question_area_layout.setSpacing(15)
        self.progress_label = QLabel("")
        self.progress_label.setObjectName("ProgressLabel")
        question_area_layout.addWidget(
            self.progress_label, alignment=Qt.AlignmentFlag.AlignRight
        )
        self.question_label = QLabel("Question text will appear here.")
        self.question_label.setObjectName("QuestionLabel")
        self.question_label.setWordWrap(True)
        question_area_layout.addWidget(self.question_label)
        self.options_layout_container = QVBoxLayout()
        self.options_layout_container.setSpacing(10)
        self.radio_button_group = QButtonGroup(self)
        self.radio_button_group.setExclusive(True)
        question_area_layout.addLayout(self.options_layout_container)
        question_area_layout.addSpacerItem(
            QSpacerItem(5, 10, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        )
        self.main_layout.addWidget(self.question_area_widget)
        self.nav_buttons_layout = QHBoxLayout()
        self.nav_buttons_layout.setSpacing(10)
        self.prev_button = QPushButton("Previous")
        self.prev_button.setObjectName("PrevButton")
        self.prev_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.prev_button.clicked.connect(self.go_previous)
        self.nav_buttons_layout.addSpacerItem(
            QSpacerItem(
                40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
            )
        )
        self.nav_buttons_layout.addWidget(self.prev_button)
        self.next_button = QPushButton("Next")
        self.next_button.setObjectName("NextButton")
        self.next_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.next_button.setDefault(True)
        self.next_button.clicked.connect(self.go_next)
        self.nav_buttons_layout.addWidget(self.next_button)
        self.main_layout.addLayout(self.nav_buttons_layout)
        self.setMinimumSize(550, 500)
        self.resize(600, 550)

    def apply_styles(self):
        # ... (QSS - same as before) ...
        qss = """
            QWidget#MainWindow { background-color: #F4F6F8; font-family: 'Segoe UI', Arial, sans-serif; }
            QLabel#DisclaimerLabel { background-color: #FFF3CD; color: #664D03; border: 1px solid #FFECB5; border-radius: 6px; padding: 12px; font-size: 9pt; font-weight: normal; }
            QLabel#InstructionsLabel { color: #4A5568; font-size: 11pt; padding-left: 5px; margin-bottom: 0px; }
            QWidget#QuestionAreaWidget { background-color: #FFFFFF; border-radius: 8px; border: 1px solid #E2E8F0; }
            QLabel#QuestionLabel { color: #1A202C; font-size: 13pt; font-weight: bold; line-height: 1.5; padding-bottom: 10px; }
            QLabel#ProgressLabel { color: #718096; font-size: 9pt; font-weight: bold; }
            QRadioButton { font-size: 11pt; color: #2D3748; padding: 8px 5px; spacing: 10px; }
            QRadioButton::indicator { width: 18px; height: 18px; }
            QRadioButton::indicator:unchecked { border: 2px solid #A0AEC0; border-radius: 9px; background-color: #FFFFFF; }
            QRadioButton::indicator:unchecked:hover { border: 2px solid #718096; }
            QRadioButton::indicator:checked { border: 2px solid #3182CE; border-radius: 9px; background-color: #3182CE; }
            /* QRadioButton::indicator:checked::after { content: ""; display: block; width: 8px; height: 8px; margin: 3px; border-radius: 4px; background-color: white; } */
            QPushButton { font-size: 11pt; font-weight: bold; padding: 10px 20px; border-radius: 6px; border: none; min-width: 100px; }
            QPushButton#NextButton { background-color: #3182CE; color: white; }
            QPushButton#NextButton:hover { background-color: #2B6CB0; }
            QPushButton#NextButton:pressed { background-color: #2C5282; }
            QPushButton#PrevButton { background-color: #E2E8F0; color: #2D3748; }
            QPushButton#PrevButton:hover { background-color: #CBD5E0; }
            QPushButton#PrevButton:pressed { background-color: #A0AEC0; }
            QPushButton:disabled { background-color: #E2E8F0; color: #A0AEC0; }
            QMessageBox { font-family: 'Segoe UI', Arial, sans-serif; font-size: 10pt; }
            QMessageBox QLabel { color: #2D3748; }
            QMessageBox QPushButton { background-color: #CBD5E0; color: #2D3748; padding: 8px 15px; border-radius: 4px; min-width: 80px; }
            QMessageBox QPushButton:hover { background-color: #A0AEC0; }
        """
        self.setStyleSheet(qss)

    def display_question(self):
        # ... (same as before) ...
        for button in self.radio_button_group.buttons():
            self.radio_button_group.removeButton(button)
            button.deleteLater()
        while self.options_layout_container.count():
            child = self.options_layout_container.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        if self.current_question_index < self.num_questions:
            question_data = self.questions[self.current_question_index]
            self.question_label.setText(question_data["text"])
            self.progress_label.setText(
                f"Question {self.current_question_index + 1} of {self.num_questions}"
            )
            for option_text in question_data["options"]:
                radio_button = QRadioButton(option_text)
                radio_button.setCursor(Qt.CursorShape.PointingHandCursor)
                self.options_layout_container.addWidget(radio_button)
                self.radio_button_group.addButton(radio_button)
                radio_button.toggled.connect(
                    lambda checked, q_idx=self.current_question_index, opt_txt=option_text: self._handle_option_toggled(
                        checked, q_idx, opt_txt
                    )
                )
                if self.user_answers[self.current_question_index] == option_text:
                    radio_button.setChecked(True)
            self.prev_button.setEnabled(self.current_question_index > 0)
            if self.current_question_index == self.num_questions - 1:
                self.next_button.setText("Finish")
            else:
                self.next_button.setText("Next")
        else:
            self.process_survey()

    def _handle_option_toggled(self, checked, question_idx, option_text):
        # ... (same as before, logs to survey_log_file_name) ...
        if checked:
            self._log_event(
                action_type="active",
                event_type="option_selected",
                details={
                    "question_index": question_idx + 1,
                    "selected_option": option_text,
                },
            )
            self.user_answers[question_idx] = option_text

    def go_next(self):
        # ... (same as before, logs to survey_log_file_name) ...
        current_q_idx_for_log = self.current_question_index + 1
        action_text = (
            "next_clicked" if self.next_button.text() == "Next" else "finish_clicked"
        )
        self._log_event(
            action_type="active",
            event_type=action_text,
            details={"from_question": current_q_idx_for_log},
        )
        if self.user_answers[self.current_question_index] is None:
            QMessageBox.warning(
                self, "No Answer", "Please select an answer before proceeding."
            )
            self._log_event(
                action_type="passive",
                event_type="validation_error",
                details={
                    "message": "No answer selected for question "
                    + str(current_q_idx_for_log)
                },
            )
            return
        if self.current_question_index < self.num_questions - 1:
            self.current_question_index += 1
            self.display_question()
            self._log_event(
                action_type="passive",
                event_type="question_displayed",
                details={"question_index": self.current_question_index + 1},
            )
        else:
            self.process_survey()

    def go_previous(self):
        # ... (same as before, logs to survey_log_file_name) ...
        current_q_idx_for_log = self.current_question_index + 1
        self._log_event(
            action_type="active",
            event_type="previous_clicked",
            details={"from_question": current_q_idx_for_log},
        )
        if self.current_question_index > 0:
            self.current_question_index -= 1
            self.display_question()
            # Corrected action_type for question_displayed on previous
            self._log_event(
                action_type="passive",
                event_type="question_displayed",
                details={"question_index": self.current_question_index + 1},
            )

    def process_survey(self):
        # ... (same as before, logs to survey_log_file_name) ...
        self._log_event(action_type="passive", event_type="survey_submitted")
        if any(answer is None for answer in self.user_answers):
            QMessageBox.warning(
                self,
                "Incomplete Survey",
                "One or more questions were not answered. Please review.",
            )
            self._log_event(
                action_type="passive",
                event_type="validation_error",
                details={"message": "Incomplete survey."},
            )
            return
        yes_count = 0
        responses_summary = []
        for i, answer in enumerate(self.user_answers):
            question_text_full = self.questions[i]["text"]
            question_text_short = (
                question_text_full.split(". ", 1)[1]
                if ". " in question_text_full
                else question_text_full
            )
            responses_summary.append(f"- {question_text_short}: {answer}")
            if answer == "Yes":
                yes_count += 1
        result_message_intro = (
            "Thank you for completing the check-in.\n\nYour responses:\n"
            + "\n".join(responses_summary)
            + "\n\n"
        )
        if yes_count >= 2:
            result_message_intro += (
                "Based on your responses, it might be helpful to talk to someone about how you're feeling. "
                + "Remember, support is available, and speaking with a qualified healthcare professional can provide guidance."
            )
        else:
            result_message_intro += (
                "Thank you for taking the time for this check-in. Remember to prioritize your well-being. "
                + "If you ever feel overwhelmed or have concerns, reaching out to a healthcare professional is a positive step."
            )
        result_message_intro += "\n\n<b>Disclaimer: This tool is for illustrative purposes only and is not a substitute for professional medical advice, diagnosis, or treatment.</b>"
        dialog = QMessageBox(self)
        dialog.setStyleSheet(self.styleSheet())
        dialog.setWindowTitle("Check-in Complete")
        dialog.setIcon(QMessageBox.Icon.Information)
        dialog.setTextFormat(Qt.TextFormat.RichText)
        dialog.setText(result_message_intro)
        dialog.exec()
        self.next_button.setEnabled(False)
        self.prev_button.setEnabled(False)
        self._log_event(action_type="passive", event_type="survey_completed")
        self.close()

    def closeEvent(self, event):
        # ... (same as before, logs to survey_log_file_name) ...
        self._stop_webcam_capture()
        self._log_event(
            action_type="passive", event_type="application_closed"
        )  # This logs to survey_log
        print(
            f"Application closed. Survey log: '{self.survey_log_file_name}', Prediction log: '{self.prediction_log_file_name}'."
        )
        event.accept()


# --- Function to Display Survey Log (Outside the App Class) ---
def display_survey_log(log_file_name):
    # ... (same as before) ...
    print(
        f"\n--- Attempting to Display Survey Interaction Log from '{log_file_name}' ---"
    )
    try:
        if not os.path.exists(log_file_name) or os.path.getsize(log_file_name) == 0:
            print(f"Log file '{log_file_name}' not found or is empty.")
            return
        with open(log_file_name, "r", encoding="utf-8") as f:
            logs = json.load(f)
        if not isinstance(logs, list) or not logs:
            print(f"No valid logs found in the JSON file: {log_file_name}.")
            return
        print(f"\nInteraction Log from '{log_file_name}':")
        print("-" * 80)
        print(f"{'Timestamp':<25} | {'Action':<8} | {'Event Type':<25} | {'Details'}")
        print("-" * 80)
        for log_entry in logs:
            ts = log_entry.get("timestamp", "N/A")
            action = log_entry.get("action_type", "N/A")
            event_type_val = log_entry.get("event_type", "N/A")
            details_dict = log_entry.get("details", {})
            details_str_parts = []
            if isinstance(details_dict, dict):
                for key, value in details_dict.items():
                    details_str_parts.append(f"{key}: {value}")
            elif details_dict:
                details_str_parts.append(str(details_dict))
            details_str = ", ".join(details_str_parts) if details_str_parts else ""
            print(f"{ts:<25} | {action:<8} | {event_type_val:<25} | {details_str}")
        print("-" * 80)
    except json.JSONDecodeError:
        print(
            f"Error: Could not decode JSON from '{log_file_name}'. The file might be corrupted."
        )
    except Exception as e:
        print(
            f"An unexpected error occurred while displaying logs from '{log_file_name}': {e}"
        )


# --- New Function to Display Prediction Log ---
def display_prediction_log(log_file_name):
    """
    Reads a specific JSON prediction log file and prints its content to the console.
    """
    print(
        f"\n--- Attempting to Display Image Prediction Log from '{log_file_name}' ---"
    )
    try:
        if not os.path.exists(log_file_name) or os.path.getsize(log_file_name) == 0:
            print(f"Prediction log file '{log_file_name}' not found or is empty.")
            return

        with open(log_file_name, "r", encoding="utf-8") as f:
            predictions = json.load(f)

        if not isinstance(predictions, list) or not predictions:
            print(f"No valid predictions found in the JSON file: {log_file_name}.")
            return

        print(f"\nImage Prediction Log from '{log_file_name}':")
        print("-" * 60)
        print(f"{'Timestamp':<25} | {'Predicted Label':<20} | {'Confidence':<10}")
        print("-" * 60)
        for entry in predictions:
            ts = entry.get("timestamp", "N/A")
            label = entry.get("predicted_label", "N/A")
            confidence = entry.get("confidence", "N/A")
            print(f"{ts:<25} | {label:<20} | {confidence:<10.4f}")
        print("-" * 60)

    except json.JSONDecodeError:
        print(
            f"Error: Could not decode JSON from prediction log '{log_file_name}'. The file might be corrupted."
        )
    except Exception as e:
        print(
            f"An unexpected error occurred while displaying prediction logs from '{log_file_name}': {e}"
        )


if __name__ == "__main__":
    run_application = True
    if run_application:
        app_instance = QApplication(sys.argv)
        survey_app = ModernMentalHealthSurveyApp()
        survey_app.show()
        exit_code = app_instance.exec()
        if exit_code == 0:
            print(f"\nApplication finished.")
            display_survey_log(survey_app.survey_log_file_name)
            display_prediction_log(
                survey_app.prediction_log_file_name
            )  # Display prediction log as well
        sys.exit(exit_code)
    else:
        # Example: Manually display logs if app is not run
        # display_survey_log("survey_log_YYYYMMDD_HHMMSS.json")
        # display_prediction_log("prediction_log_YYYYMMDD_HHMMSS_ffffff.json")
        print("Set run_application = True to run the survey.")
        sys.exit(0)
