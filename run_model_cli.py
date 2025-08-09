#!/usr/bin/env python3
"""
CLI entry point for Instagram Engagement ML model training and evaluation
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from models.model_trainer import main
import argparse
import os
import json
from src.models.model_trainer import ModelTrainer

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate Instagram engagement prediction model.")
    parser.add_argument('--owner_id', type=str, default=None, help='Instagram account owner_id')
    parser.add_argument('--model_type', type=str, default='random_forest', choices=['random_forest','xgboost','lightgbm','tabnet','gnn','bert'], help='Model type to train')
    parser.add_argument('--use_sentiment', action='store_true', help='Include sentiment_weighted_engagement feature')
    parser.add_argument('--use_hashtags', action='store_true', help='Include hashtags features')
    parser.add_argument('--use_emoji', action='store_true', help='Include emoji features')
    parser.add_argument('--cv', action='store_true', help='Enable cross-validation')
    parser.add_argument('--output_path', type=str, default='outputs/', help='Output directory for results')
    parser.add_argument('--target', type=str, default=None, help='Target variable for training (e.g. engagement_rate, sentiment_weighted_engagement, etc)')
    args = parser.parse_args()

    trainer = ModelTrainer()
    # Prepare features list based on toggles
    feature_toggles = {
        'sentiment_weighted_engagement': args.use_sentiment,
        'hashtags': args.use_hashtags,
        'emoji': args.use_emoji
    }
    # Determine target
    if args.target:
        target = args.target
    elif args.use_sentiment:
        target = 'sentiment_weighted_engagement'
    else:
        target = 'engagement_rate'
    
    # Map CLI model_type to internal model name
    model_type_map = {
        'random_forest': 'Random Forest',
        'xgboost': 'XGBoost',
        'lightgbm': 'LightGBM',
        'tabnet': 'TabNet',
        'gnn': 'GNN',
        'bert': 'BERT',
    }
    internal_model_name = model_type_map.get(args.model_type)
    if not internal_model_name:
        print(f"[ERROR] Unknown model_type: {args.model_type}")
        sys.exit(1)

    print(f"[INFO] Training {args.model_type} for owner_id={args.owner_id} (target={target}) ...")
    results = trainer.train_engagement_model(
        owner_id=args.owner_id,
        models=[internal_model_name],
        target=target,
        test_size=0.2,
        cv_folds=5 if args.cv else 1,
        output_path=args.output_path,
        feature_toggles=feature_toggles
    )
    print("[INFO] Training complete. Results:")
    # Print accuracy metrics for each model
    any_success = False
    for model, metrics in results.items():
        if isinstance(metrics, dict) and "accuracy" in metrics:
            any_success = True
            print(f"{model}: Accuracy={metrics.get('accuracy', 0):.3f}, F1={metrics.get('f1_score', 0):.3f}, Precision={metrics.get('precision', 0):.3f}, Recall={metrics.get('recall', 0):.3f}, ROC-AUC={metrics.get('roc_auc', 0):.3f}")
            if 'top_features' in metrics:
                print(f"  Top features: {metrics['top_features']}")
                if 'sentiment_weighted_engagement' in metrics['top_features']:
                    print("  [INFO] sentiment_weighted_engagement is important for this model!")
        elif isinstance(metrics, dict) and "error" in metrics:
            print(f"{model}: ERROR: {metrics['error']}")
    if not any_success:
        print("[WARN] No models were successfully trained. Check logs and data.")
    # Show recommendations
    try:
        with open(os.path.join(args.output_path, 'guidelines.json')) as f:
            guidelines = json.load(f)
        print("\n[INFO] Recommended Keywords:", guidelines.get('recommended_keywords', []))
        print("[INFO] Recommended Caption Phrases:", guidelines.get('caption_phrases', []))
        print("[INFO] Recommended Hashtags:", guidelines.get('recommended_hashtags', []))
    except Exception:
        pass

if __name__ == "__main__":
    main()
