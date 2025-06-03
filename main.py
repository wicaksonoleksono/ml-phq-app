import sys
import json
import threading
import time
from typing import Literal
import cv2
import numpy as np  # For image manipulation and argmax
import onnxruntime  # For ONNX model inference
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
    # QFrame,
    QSpacerItem,
    QSizePolicy,
)
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import QScreen

from datetime import datetime
import os


# --- PyQt6 Application Class ---
class ModernMentalHealthSurveyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mental Health Quick Check")
        # self.setGeometry(150, 150, 600, 550)

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

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file_name = f"survey_log_{timestamp_str}.json"
        print(f"Current session log file: {self.log_file_name}")

        self._log_event(action_type="passive", event_type="app_init")

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
        )

    def _load_onnx_model(self):
        if not os.path.exists(self.onnx_model_path):
            print(
                f"ONNX Model Error: File not found at '{self.onnx_model_path}'. Inference will be disabled."
            )
            # You could show a QMessageBox here if QApplication instance already exists
            # For now, just printing.
            self.ort_session = None
            return
        try:
            self.ort_session = onnxruntime.InferenceSession(self.onnx_model_path)
            self.input_name = self.ort_session.get_inputs()[0].name
            # Assuming the first output is the one with class scores/logits
            self.output_name = self.ort_session.get_outputs()[0].name
            print(f"ONNX model '{self.onnx_model_path}' loaded successfully.")
            print(
                f"Model Input Name: {self.input_name}, Output Name: {self.output_name}"
            )
        except Exception as e:
            print(f"Error loading ONNX model: {e}")
            self.ort_session = None  # Disable inference if model loading fails

    def _center_window(self):
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
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_entry = {
            "timestamp": timestamp,
            "action_type": action_type,
            "event_type": event_type,
        }
        if details:  # If details are provided, add them directly
            log_entry["details"] = details

        all_logs = []
        try:
            if (
                os.path.exists(self.log_file_name)
                and os.path.getsize(self.log_file_name) > 0
            ):
                with open(self.log_file_name, "r", encoding="utf-8") as f:
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
            with open(self.log_file_name, "w", encoding="utf-8") as f:
                json.dump(all_logs, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Error writing to log file '{self.log_file_name}': {e}")

    def _preprocess_image(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocesses an image frame for ONNX model inference.
        Assumes common preprocessing: BGR_to_RGB, resize, normalize [0,1], HWC_to_CHW, batch.
        Adjust this function based on your specific model's requirements.
        """
        # 1. Resize
        img = cv2.resize(frame, self.input_size)  # self.input_size is (height, width)

        # 2. BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 3. Convert to float32 and scale to [0, 1]
        img = img.astype(np.float32) / 255.0

        # 4. Normalize with mean and std
        # These are typically for RGB channels
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # Ensure mean and std are broadcastable to img (H, W, C)
        # img shape is (height, width, 3) at this point
        img = (img - mean) / std

        # 5. HWC to CHW (Height, Width, Channel to Channel, Height, Width)
        img = np.transpose(img, (2, 0, 1))  # C, H, W

        # 6. Add batch dimension (1, C, H, W)
        img = np.expand_dims(img, axis=0)
        return img

    def _start_webcam_capture(self):
        if self.capture_thread is None:
            self.capture_active = True
            self.capture_thread = threading.Thread(
                target=self._webcam_capture_loop, daemon=True
            )
            self.capture_thread.start()
            print("Webcam capture thread started.")

    def _webcam_capture_loop(self):
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
                    # print(f"{datetime.now()}: Webcam frame captured successfully.") # Optional: reduce console spam
                    if self.ort_session:  # Perform inference if model is loaded
                        try:
                            preprocessed_frame = self._preprocess_image(
                                frame.copy()
                            )  # Use a copy

                            ort_inputs = {self.input_name: preprocessed_frame}
                            ort_outs = self.ort_session.run(
                                [self.output_name], ort_inputs
                            )

                            # Assuming ort_outs[0] is the scores/logits for classification
                            # Output shape might be (1, num_classes)
                            scores = ort_outs[0][
                                0
                            ]  # Get the scores for the single image in the batch
                            predicted_index = np.argmax(scores)
                            confidence = float(
                                scores[predicted_index]
                            )  # Or use softmax if output is logits

                            predicted_label = "Unknown"
                            if 0 <= predicted_index < len(self.class_labels):
                                predicted_label = self.class_labels[predicted_index]

                            print(
                                f"{datetime.now()}: Predicted Label: {predicted_label} (Confidence: {confidence:.2f})"
                            )

                            self._log_event(
                                action_type="passive",
                                event_type="image_prediction",
                                details={
                                    "predicted_label": predicted_label,
                                    "confidence": round(confidence, 4),
                                    "predicted_index": int(predicted_index),
                                    # Optional: "all_scores": [round(float(s), 4) for s in scores]
                                },
                            )
                        except Exception as e:
                            print(
                                f"{datetime.now()}: Error during model inference: {e}"
                            )
                else:
                    print(
                        f"{datetime.now()}: Error: Failed to capture frame from webcam."
                    )

                for _ in range(
                    10
                ):  # Check active flag frequently for responsive shutdown
                    if not self.capture_active:
                        break
                    time.sleep(0.1)  # Total ~1 second effective sleep
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
        print("Attempting to stop webcam capture thread...")
        self.capture_active = False
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.5)  # Increased timeout slightly
            if self.capture_thread.is_alive():
                print("Webcam capture thread did not stop in time.")
            else:
                print("Webcam capture thread stopped.")
        self.capture_thread = None

    def init_ui(self):
        # ... (UI setup - same as your previous ModernMentalHealthSurveyApp) ...
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
        # ... (QSS - same as your previous ModernMentalHealthSurveyApp) ...
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
            QRadioButton::indicator:checked::after { content: ""; display: block; width: 8px; height: 8px; margin: 3px; border-radius: 4px; background-color: white; }
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
        # ... (same as your previous ModernMentalHealthSurveyApp) ...
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
        # ... (same as your previous ModernMentalHealthSurveyApp) ...
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
        # ... (same as your previous ModernMentalHealthSurveyApp) ...
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
        # ... (same as your previous ModernMentalHealthSurveyApp) ...
        current_q_idx_for_log = self.current_question_index + 1
        self._log_event(
            action_type="active",
            event_type="previous_clicked",
            details={"from_question": current_q_idx_for_log},
        )
        if self.current_question_index > 0:
            self.current_question_index -= 1
            self.display_question()
            self._log_event(
                action_type="active",
                event_type="question_displayed",
                details={"question_index": self.current_question_index + 1},
            )  # Should be passive

    def process_survey(self):
        # ... (same as your previous ModernMentalHealthSurveyApp) ...
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
        # ... (same as your previous ModernMentalHealthSurveyApp) ...
        self._stop_webcam_capture()
        self._log_event(action_type="passive", event_type="application_closed")
        print(
            f"Application closed. Log for this session saved to '{self.log_file_name}'."
        )
        event.accept()


# --- Function to Display Log (Outside the App Class) ---
def display_survey_log(log_file_name):
    # ... (same as your previous ModernMentalHealthSurveyApp) ...
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


if __name__ == "__main__":
    # ... (same as your previous ModernMentalHealthSurveyApp) ...
    run_application = True
    if run_application:
        app_instance = QApplication(sys.argv)
        survey_app = ModernMentalHealthSurveyApp()
        survey_app.show()
        exit_code = app_instance.exec()
        if exit_code == 0:
            print(
                f"\nApplication finished. Displaying log for session: {survey_app.log_file_name}"
            )
            display_survey_log(survey_app.log_file_name)
        sys.exit(exit_code)
    else:
        print(
            "To display logs, please specify a log file name or run the application with run_application = True."
        )
        sys.exit(0)
