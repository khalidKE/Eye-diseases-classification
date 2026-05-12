#!/usr/bin/env python3
"""
Main Entry Point for Eye Disease Classification Project

Usage:
    # Data preprocessing
    python main.py preprocess
    
    # Train a single model
    python main.py train --model cnn_baseline
    
    # Train all models
    python main.py train --all
    
    # Evaluate a model
    python main.py evaluate --model models/cnn_baseline/final_model.h5
    
    # Predict on single image
    python main.py predict --image data/test/image.jpg --model models/best_model.h5
    
    # Run web app
    python main.py app
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import get_config
from src.utils import print_system_info, set_random_seed


def preprocess_command(args):
    """Run data preprocessing."""
    from src.preprocessing import process_dataset
    from src.config import get_config
    
    print_system_info()
    print("\n" + "="*60)
    print("DATA PREPROCESSING")
    print("="*60)
    
    config = get_config()
    
    raw_dir = args.raw_dir or config.get('dataset.raw_data_path', 'data/raw')
    processed_dir = args.processed_dir or config.get('dataset.processed_data_path', 'data/processed')
    classes = config.get('dataset.classes', ['normal', 'diabetic_retinopathy', 'cataract', 'glaucoma'])
    image_size = args.image_size or config.get('dataset.image_size', 224)
    
    print(f"Raw data: {raw_dir}")
    print(f"Processed data: {processed_dir}")
    print(f"Classes: {classes}")
    print(f"Image size: {image_size}")
    print("="*60 + "\n")
    
    if not os.path.exists(raw_dir):
        print(f"❌ Error: Raw data directory not found: {raw_dir}")
        print("\nPlease download the dataset from:")
        print("https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification")
        return
    
    stats = process_dataset(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        classes=classes,
        target_size=(image_size, image_size)
    )
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE")
    print("="*60)
    print(f"Total images: {stats['total_images']}")
    print(f"Successfully processed: {stats['processed']}")
    print(f"Corrupt/invalid: {stats['corrupt']}")


def train_command(args):
    """Run model training."""
    from src.train import train_model, train_all_models
    from src.config import get_config
    
    print_system_info()
    print("\n" + "="*60)
    print("MODEL TRAINING")
    print("="*60)
    
    # Set random seed for reproducibility
    set_random_seed(42)
    
    config = get_config()
    data_dir = config.get('dataset.processed_data_path', 'data/processed')
    classes = config.get('dataset.classes', ['normal', 'diabetic_retinopathy', 'cataract', 'glaucoma'])
    
    if not os.path.exists(data_dir):
        print(f"❌ Error: Processed data not found: {data_dir}")
        print("\nPlease run preprocessing first:")
        print("  python main.py preprocess")
        return
    
    training_config = {
        'batch_size': args.batch_size or 32,
        'epochs': args.epochs or 50,
        'learning_rate': args.learning_rate or 0.001,
        'optimizer': args.optimizer or 'adam'
    }
    
    if args.all:
        # Train all models
        print("Training all models...\n")
        models_to_train = ['cnn_baseline', 'resnet50', 'efficientnet']
        results = train_all_models(
            data_dir=data_dir,
            classes=classes,
            models=models_to_train,
            training_config=training_config
        )
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        for model_name, result in results.items():
            if result:
                model, history = result
                final_acc = history.history['val_accuracy'][-1]
                print(f"{model_name}: {final_acc:.4f} validation accuracy")
    
    else:
        # Train single model
        model_name = args.model or 'cnn_baseline'
        print(f"Training model: {model_name}\n")
        
        model, history = train_model(
            model_name=model_name,
            data_dir=data_dir,
            classes=classes,
            training_config=training_config
        )
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        final_acc = history.history['val_accuracy'][-1]
        final_loss = history.history['val_loss'][-1]
        print(f"Final validation accuracy: {final_acc:.4f}")
        print(f"Final validation loss: {final_loss:.4f}")


def evaluate_command(args):
    """Run model evaluation."""
    from src.evaluation import evaluate_model, print_metrics_report, save_evaluation_report
    from src.evaluation import plot_confusion_matrix, plot_roc_curves, plot_training_history
    from tensorflow import keras
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from src.config import get_config
    
    print_system_info()
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    if not args.model:
        print("❌ Error: Please specify a model path with --model")
        return
    
    if not os.path.exists(args.model):
        print(f"❌ Error: Model not found: {args.model}")
        return
    
    config = get_config()
    data_dir = config.get('dataset.processed_data_path', 'data/processed')
    classes = config.get('dataset.classes', ['normal', 'diabetic_retinopathy', 'cataract', 'glaucoma'])
    
    # Check for test split folder
    test_dir = os.path.join(data_dir, 'test')
    if not os.path.exists(test_dir):
        print(f"❌ Error: Test folder not found: {test_dir}")
        print("Please ensure data is split into train/val/test folders.")
        return
    
    print(f"Loading model: {args.model}")
    model = keras.models.load_model(args.model)
    
    print("Loading test data...")
    print(f"Test directory: {test_dir}")
    
    # Create test generator from test folder
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=32,
        classes=classes,
        class_mode='categorical',
        shuffle=False
    )
    
    print(f"Found {test_gen.samples} test samples")
    print("Evaluating model...")
    metrics = evaluate_model(model, test_gen, classes)
    
    # Print report
    print_metrics_report(metrics, classes)
    
    # Save report
    model_name = Path(args.model).stem
    save_evaluation_report(metrics, model_name)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Collect predictions batch by batch (Keras 3 compatible)
    print("Collecting predictions for visualizations...")
    y_pred_prob = []
    y_true = []
    
    num_batches = 0
    max_batches = (test_gen.samples // test_gen.batch_size) + 1
    
    for batch_x, batch_y in test_gen:
        batch_pred = model.predict(batch_x, verbose=0)
        y_pred_prob.append(batch_pred)
        y_true.extend(batch_y.argmax(axis=1))
        
        num_batches += 1
        if num_batches >= max_batches:
            break
    
    y_pred_prob = np.concatenate(y_pred_prob, axis=0)
    y_pred = y_pred_prob.argmax(axis=1)
    y_true = np.array(y_true)
    
    plot_confusion_matrix(
        y_true, y_pred, classes,
        save_path=f'results/{model_name}_confusion_matrix.png'
    )
    
    plot_roc_curves(
        y_true, y_pred_prob, classes,
        save_path=f'results/{model_name}_roc_curves.png'
    )
    
    print(f"\n✅ Results saved to 'results/{model_name}_*'")


def predict_command(args):
    """Run prediction on single image."""
    from src.predict import predict_single_image, format_prediction_result
    
    print("="*60)
    print("PREDICTION")
    print("="*60)
    
    if not args.image or not args.model:
        print("❌ Error: Please specify both --image and --model")
        return
    
    if not os.path.exists(args.image):
        print(f"❌ Error: Image not found: {args.image}")
        return
    
    if not os.path.exists(args.model):
        print(f"❌ Error: Model not found: {args.model}")
        return
    
    result = predict_single_image(
        model_path=args.model,
        image_path=args.image
    )
    
    print("\n" + format_prediction_result(result, detailed=True))


def app_command(args):
    """Run Streamlit web application."""
    import subprocess
    
    print("="*60)
    print("STARTING WEB APPLICATION")
    print("="*60)
    print("\nThe app will open in your browser.")
    print("Press Ctrl+C to stop the server.\n")
    
    try:
        subprocess.run(
            ["streamlit", "run", "src/app.py"],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting app: {e}")
    except KeyboardInterrupt:
        print("\n👋 App stopped")


def main():
    """Main function to parse arguments and run commands."""
    parser = argparse.ArgumentParser(
        description="Eye Disease Classification - Main Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preprocess data
  python main.py preprocess
  
  # Train a model
  python main.py train --model resnet50 --epochs 30
  
  # Train all models
  python main.py train --all
  
  # Evaluate model
  python main.py evaluate --model models/resnet50/final_model.h5
  
  # Predict single image
  python main.py predict --image test.jpg --model models/best_model.h5
  
  # Run web app
  python main.py app
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Preprocess command
    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocess raw images')
    preprocess_parser.add_argument('--raw-dir', help='Raw data directory')
    preprocess_parser.add_argument('--processed-dir', help='Processed data directory')
    preprocess_parser.add_argument('--image-size', type=int, default=224, help='Target image size')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train models')
    train_parser.add_argument('--model', choices=['cnn_baseline', 'resnet50', 'efficientnet'],
                             help='Model architecture to train')
    train_parser.add_argument('--all', action='store_true', help='Train all models')
    train_parser.add_argument('--epochs', type=int, help='Number of epochs')
    train_parser.add_argument('--batch-size', type=int, help='Batch size')
    train_parser.add_argument('--learning-rate', type=float, help='Learning rate')
    train_parser.add_argument('--optimizer', choices=['adam', 'sgd', 'rmsprop'], help='Optimizer')
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate a trained model')
    evaluate_parser.add_argument('--model', required=True, help='Path to model file')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict on single image')
    predict_parser.add_argument('--image', required=True, help='Path to image file')
    predict_parser.add_argument('--model', required=True, help='Path to model file')
    
    # App command
    app_parser = subparsers.add_parser('app', help='Run web application')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Run appropriate command
    commands = {
        'preprocess': preprocess_command,
        'train': train_command,
        'evaluate': evaluate_command,
        'predict': predict_command,
        'app': app_command
    }
    
    command_func = commands.get(args.command)
    if command_func:
        command_func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
