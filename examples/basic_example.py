"""
Example notebook for testing the pipeline
"""
import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from src.pipeline import TradingPipeline
from src.logger import get_logger

logger = get_logger(__name__)

def main():
    """Run example pipeline"""
    logger.info("starting_example_pipeline")
    
    pipeline = TradingPipeline()
    
    # Step 1: Fetch data
    logger.info("step_1_fetching_data")
    try:
        # Fetch last 10 days of BTC data
        long_data, short_data, full_df = pipeline.fetch_and_prepare_data(
            "BTCUSDT",
            days_back=10,
            lookback_days=5
        )
        logger.info("data_fetched_successfully", shape=full_df.shape)
        print(f"✓ Data fetched: {full_df.shape[0]} candles")
    except Exception as e:
        logger.error("data_fetch_failed", error=str(e))
        print(f"✗ Data fetch failed: {e}")
        return
    
    # Step 2: Train MacroNet
    logger.info("step_2_training_macronet")
    try:
        # Train on a small dataset for testing
        sample_features = pipeline.extract_feature_arrays(long_data)
        X = sample_features[np.newaxis, :, :]
        
        pipeline.macronet.train(X, epochs=5)
        logger.info("macronet_trained")
        print("✓ MacroNet trained")
    except Exception as e:
        logger.error("macronet_training_failed", error=str(e))
        print(f"✗ MacroNet training failed: {e}")
        return
    
    # Step 3: Generate macro embedding
    logger.info("step_3_generating_embedding")
    try:
        macro_emb = pipeline.generate_macro_embedding("BTCUSDT", days_back=5)
        logger.info("embedding_generated", shape=macro_emb.shape)
        print(f"✓ Macro embedding generated: {macro_emb.shape}")
    except Exception as e:
        logger.error("embedding_generation_failed", error=str(e))
        print(f"✗ Embedding generation failed: {e}")
        return
    
    # Step 4: Generate signal
    logger.info("step_4_generating_signal")
    try:
        signal = pipeline.predict_signal("BTCUSDT")
        logger.info("signal_generated", value=signal)
        print(f"✓ Signal generated: {signal:.3f}")
    except Exception as e:
        logger.error("signal_generation_failed", error=str(e))
        print(f"✗ Signal generation failed: {e}")
    
    logger.info("example_pipeline_complete")
    print("\n✓ Example pipeline complete!")

if __name__ == "__main__":
    main()
