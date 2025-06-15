python preprocess_features.py --input ./Data.facial --output ./runs/
python train_classfier.py --csv_path runs/delta_emotion_features.csv
python export_onnx.py \
 --model_path runs/best_emotion_classifier.pth \
 --csv_path runs/delta_emotion_features.csv \
 --output_path runs/emotion_model.onnx
python baseline.py
python camera.py
