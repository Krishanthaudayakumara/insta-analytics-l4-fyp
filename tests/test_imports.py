#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(__file__))

try:
    from src.preprocessing.data_processor import DataProcessor
    print("✅ DataProcessor imported successfully")
except Exception as e:
    print(f"❌ Error importing DataProcessor: {e}")

try:
    from src.follower_selection.high_value_selector import HighValueFollowerSelector
    print("✅ HighValueFollowerSelector imported successfully")
except Exception as e:
    print(f"❌ Error importing HighValueFollowerSelector: {e}")

try:
    from src.sentiment_analysis.bert_analyzer import BERTSentimentAnalyzer
    print("✅ BERTSentimentAnalyzer imported successfully")
except Exception as e:
    print(f"❌ Error importing BERTSentimentAnalyzer: {e}")

try:
    from src.models.model_trainer import ModelTrainer
    print("✅ ModelTrainer imported successfully")
except Exception as e:
    print(f"❌ Error importing ModelTrainer: {e}")

try:
    from src.evaluation.model_evaluator import ModelEvaluator
    print("✅ ModelEvaluator imported successfully")
except Exception as e:
    print(f"❌ Error importing ModelEvaluator: {e}")

try:
    from src.profiling.profile_generator import ProfileGenerator
    print("✅ ProfileGenerator imported successfully")
except Exception as e:
    print(f"❌ Error importing ProfileGenerator: {e}")

print("\n🚀 All basic imports completed!")
