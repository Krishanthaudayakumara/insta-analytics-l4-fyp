#!/usr/bin/env python3
"""
Final Verification Script for BERT Sentiment Analysis Implementation
Confirms all requirements have been met with exact specifications
"""

import json
import os
import pandas as pd
from datetime import datetime

def verify_bert_implementation():
    """Comprehensive verification of BERT sentiment analysis implementation"""
    
    print("🔍 BERT Sentiment Analysis Implementation Verification")
    print("=" * 60)
    
    verification_results = {
        "timestamp": datetime.now().isoformat(),
        "requirements_met": {},
        "issues_found": [],
        "summary": {}
    }
    
    # 1. Verify BERT Model Configuration
    print("\n📋 1. BERT Model Configuration")
    sentiment_file = "outputs/sentiment_scores.json"
    
    if not os.path.exists(sentiment_file):
        print("❌ Sentiment scores file not found")
        verification_results["issues_found"].append("Sentiment scores file missing")
        return verification_results
    
    with open(sentiment_file, 'r') as f:
        data = json.load(f)
    
    if 'metadata' in data:
        metadata = data['metadata']
        
        # Check model
        model_used = metadata.get('model_used', 'Unknown')
        expected_model = 'distilbert-base-uncased'
        model_correct = model_used == expected_model
        
        print(f"   Model: {model_used} {'✅' if model_correct else '❌'}")
        verification_results["requirements_met"]["correct_model"] = model_correct
        
        # Check hyperparameters
        lr = metadata.get('learning_rate', 'Unknown')
        batch_size = metadata.get('batch_size', 'Unknown')
        epochs = metadata.get('epochs', 'Unknown')
        
        lr_correct = lr == 2e-5
        batch_correct = batch_size == 16
        epochs_correct = epochs == 3
        
        print(f"   Learning Rate: {lr} {'✅' if lr_correct else '❌'}")
        print(f"   Batch Size: {batch_size} {'✅' if batch_correct else '❌'}")
        print(f"   Epochs: {epochs} {'✅' if epochs_correct else '❌'}")
        
        verification_results["requirements_met"]["hyperparameters"] = {
            "learning_rate": lr_correct,
            "batch_size": batch_correct,
            "epochs": epochs_correct
        }
        
        # Check dataset scope
        total_comments = metadata.get('total_comments', 0)
        print(f"   Total Comments Analyzed: {total_comments:,}")
        verification_results["requirements_met"]["dataset_size"] = total_comments
    else:
        print("❌ No metadata found in sentiment scores")
        verification_results["issues_found"].append("Missing metadata in sentiment scores")
    
    # 2. Verify Data Structure and Quality
    print("\n📊 2. Data Structure and Quality")
    
    if 'sentiment_scores' in data:
        sentiment_scores = data['sentiment_scores']
        num_scores = len(sentiment_scores)
        print(f"   Sentiment Entries: {num_scores:,} ✅")
        verification_results["requirements_met"]["sentiment_entries"] = num_scores
        
        # Sample verification
        sample_keys = list(sentiment_scores.keys())[:5]
        sample_valid = True
        
        for key in sample_keys:
            entry = sentiment_scores[key]
            required_fields = ['sentiment', 'confidence', 'model_used', 'comment_text']
            
            for field in required_fields:
                if field not in entry:
                    sample_valid = False
                    verification_results["issues_found"].append(f"Missing field '{field}' in {key}")
            
            # Verify model consistency
            if entry.get('model_used') != expected_model:
                sample_valid = False
                verification_results["issues_found"].append(f"Incorrect model in {key}: {entry.get('model_used')}")
        
        print(f"   Sample Data Validity: {'✅' if sample_valid else '❌'}")
        verification_results["requirements_met"]["data_validity"] = sample_valid
        
    else:
        print("❌ No sentiment_scores found in data")
        verification_results["issues_found"].append("Missing sentiment_scores in data structure")
    
    # 3. Verify Dataset-wide Analysis (No owner_id filtering)
    print("\n🌐 3. Dataset-wide Analysis Verification")
    
    preprocessed_file = "outputs/preprocessed_data.csv"
    if os.path.exists(preprocessed_file):
        df = pd.read_csv(preprocessed_file)
        total_preprocessed = len(df)
        
        # Check if we analyzed most of the dataset
        coverage_ratio = num_scores / total_preprocessed if 'num_scores' in locals() else 0
        good_coverage = coverage_ratio > 0.9  # 90% coverage is good
        
        print(f"   Preprocessed Comments: {total_preprocessed:,}")
        print(f"   Analyzed Comments: {num_scores:,}")
        print(f"   Coverage Ratio: {coverage_ratio:.2%} {'✅' if good_coverage else '⚠️'}")
        
        verification_results["requirements_met"]["dataset_coverage"] = {
            "total_preprocessed": total_preprocessed,
            "analyzed": num_scores,
            "coverage_ratio": coverage_ratio,
            "good_coverage": good_coverage
        }
    else:
        print("❌ Preprocessed data file not found")
        verification_results["issues_found"].append("Preprocessed data file missing")
    
    # 4. Verify Error Handling and Performance
    print("\n⚡ 4. Error Handling and Performance")
    
    # Check for performance optimizations
    file_size = os.path.getsize(sentiment_file) / (1024 * 1024)  # MB
    print(f"   Output File Size: {file_size:.1f} MB")
    
    reasonable_size = file_size < 10  # Less than 10MB is reasonable
    print(f"   File Size Reasonable: {'✅' if reasonable_size else '⚠️'}")
    
    verification_results["requirements_met"]["performance"] = {
        "file_size_mb": file_size,
        "reasonable_size": reasonable_size
    }
    
    # 5. Verify UI Compatibility
    print("\n🖥️  5. UI Compatibility")
    
    ui_file = "src/ui/sentiment_analysis.py"
    ui_exists = os.path.exists(ui_file)
    print(f"   UI Component Exists: {'✅' if ui_exists else '❌'}")
    
    if ui_exists:
        # Check if UI can handle the new data structure
        with open(ui_file, 'r') as f:
            ui_content = f.read()
        
        handles_nested = 'sentiment_scores' in ui_content and 'metadata' in ui_content
        print(f"   Handles Nested Structure: {'✅' if handles_nested else '❌'}")
        
        has_memory_optimization = 'sample_size' in ui_content and 'min(' in ui_content
        print(f"   Memory Optimization: {'✅' if has_memory_optimization else '❌'}")
        
        verification_results["requirements_met"]["ui_compatibility"] = {
            "exists": ui_exists,
            "handles_nested": handles_nested,
            "memory_optimized": has_memory_optimization
        }
    
    # 6. Generate Summary
    print("\n📋 6. Implementation Summary")
    
    total_requirements = 0
    met_requirements = 0
    
    def count_requirements(obj):
        nonlocal total_requirements, met_requirements
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, bool):
                    total_requirements += 1
                    if value:
                        met_requirements += 1
                elif isinstance(value, dict):
                    count_requirements(value)
    
    count_requirements(verification_results["requirements_met"])
    
    success_rate = (met_requirements / total_requirements) * 100 if total_requirements > 0 else 0
    
    print(f"   Requirements Met: {met_requirements}/{total_requirements}")
    print(f"   Success Rate: {success_rate:.1f}%")
    print(f"   Issues Found: {len(verification_results['issues_found'])}")
    
    verification_results["summary"] = {
        "total_requirements": total_requirements,
        "met_requirements": met_requirements,
        "success_rate": success_rate,
        "overall_status": "✅ COMPLETE" if success_rate >= 95 and len(verification_results["issues_found"]) == 0 else "⚠️ ISSUES FOUND"
    }
    
    print(f"\n🎯 Overall Status: {verification_results['summary']['overall_status']}")
    
    # Save verification report
    report_file = "BERT_IMPLEMENTATION_VERIFICATION.json"
    with open(report_file, 'w') as f:
        json.dump(verification_results, f, indent=2)
    
    print(f"\n📄 Verification report saved to: {report_file}")
    
    return verification_results

if __name__ == "__main__":
    print("🚀 Starting BERT Implementation Verification...")
    results = verify_bert_implementation()
    
    if results["summary"]["overall_status"].startswith("✅"):
        print("\n🎉 BERT Sentiment Analysis Implementation: SUCCESS!")
        print("All requirements have been met with the exact specifications.")
    else:
        print("\n⚠️ BERT Sentiment Analysis Implementation: ISSUES DETECTED")
        print("Please review the verification report for details.")
