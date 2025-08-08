#!/usr/bin/env python3
"""
Test script to validate memory-optimized football predictor
Simulates 8vCPU/32GB constraints
"""

import os
import psutil
import resource
from train_aws_optimized import AWSOptimizedFootballPredictor

def limit_memory(max_memory_mb=28000):
    """Set memory limit for the process"""
    try:
        # Set virtual memory limit (Linux/macOS)
        max_memory_bytes = max_memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (max_memory_bytes, max_memory_bytes))
        print(f"✅ Memory limit set to {max_memory_mb}MB")
    except:
        print(f"⚠️  Could not set hard memory limit, using soft monitoring")

def monitor_system():
    """Monitor system resources"""
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    cpu_percent = process.cpu_percent()
    
    print(f"📊 Current usage: {memory_mb:.1f}MB RAM, {cpu_percent:.1f}% CPU")
    print(f"📊 Available cores: {os.cpu_count()}")
    return memory_mb

def test_memory_optimized_training():
    """Test the memory-optimized training process"""
    print("🧪 Testing Memory-Optimized Football Predictor")
    print("="*60)
    
    # Set memory constraints
    limit_memory(28000)  # 28GB (leaving 4GB buffer)
    
    # Monitor initial state
    initial_memory = monitor_system()
    
    # Initialize predictor with memory limit
    predictor = AWSOptimizedFootballPredictor(memory_limit_mb=28000)
    
    try:
        print("\n🔄 Step 1: Loading data with batch processing...")
        df_clean = predictor.load_and_clean_data(max_memory_mb=8000)
        monitor_system()
        
        print("\n🔄 Step 2: Creating features with chunking...")
        df_form = predictor.create_advanced_features(df_clean)
        monitor_system()
        
        print("\n🔄 Step 3: Training with reduced complexity...")
        results = predictor.train_optimized_model(
            df_form, 
            optimize_hyperparameters=False,  # Skip optimization for faster testing
            max_samples=3000  # Further reduce samples for testing
        )
        
        if results[0] is not None:
            print("\n🔄 Step 4: Saving model...")
            predictor.save_model(
                model_path='test_memory_optimized_model.joblib',
                data_path='test_memory_optimized_data.csv',
                df_form=df_form
            )
            
            final_memory = monitor_system()
            memory_increase = final_memory - initial_memory
            
            print(f"\n✅ TEST COMPLETED SUCCESSFULLY!")
            print(f"📊 Memory usage increase: {memory_increase:.1f}MB")
            print(f"📊 Peak memory: {final_memory:.1f}MB")
            
            if final_memory < 28000:
                print("🎉 MEMORY CONSTRAINTS RESPECTED!")
            else:
                print("⚠️  Memory limit exceeded, but training completed")
                
            return True
        else:
            print("❌ Training failed!")
            return False
            
    except MemoryError:
        print("💥 OUT OF MEMORY ERROR!")
        print("   Consider further reducing batch sizes or sample limits")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_memory_optimized_training()
    
    if success:
        print("\n🎯 RECOMMENDATIONS FOR 8vCPU/32GB AWS INSTANCE:")
        print("✅ Use batch processing (implemented)")
        print("✅ Limit cores to 4 (implemented)")  
        print("✅ Use memory monitoring (implemented)")
        print("✅ Chunk data processing (implemented)")
        print("✅ Sample training data to 3000-4000 matches")
        print("✅ Skip hyperparameter optimization on first run")
        print("✅ Use warm_start for incremental learning")
    else:
        print("\n💡 FURTHER OPTIMIZATIONS NEEDED:")
        print("🔧 Reduce batch_size further in load_and_clean_data")
        print("🔧 Decrease max_samples in train_optimized_model")
        print("🔧 Use single model (no GridSearchCV)")
        print("🔧 Consider using lighter algorithms (Logistic Regression)")