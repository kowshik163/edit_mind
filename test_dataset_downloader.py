#!/usr/bin/env python3
"""
Test script for the enhanced dataset downloader with real annotation processing
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from utils.dataset_downloader import DatasetDownloader
import logging

def test_dataset_downloader():
    """Test the dataset downloader with real annotation processing"""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("🧪 Testing Enhanced Dataset Downloader")
    
    # Create downloader
    data_dir = Path("./test_data")
    downloader = DatasetDownloader(data_root=data_dir)
    
    logger.info(f"📁 Data directory: {data_dir.absolute()}")
    
    # Test 1: Check available datasets
    logger.info("\n📋 Available datasets:")
    for name, config in downloader.dataset_configs.items():
        logger.info(f"  • {name}: {config['name']}")
    
    # Test 2: Download a small dataset (limit samples)
    logger.info("\n📥 Testing dataset download with real annotations...")
    
    try:
        # Download just one dataset for testing
        results = downloader.download_all_datasets(
            datasets=["tvsum"],  # Start with TVSum
            force_download=False
        )
        
        if results:
            logger.info(f"✅ Download successful: {len(results)} datasets processed")
            
            for dataset_name, result in results.items():
                samples = result.get("samples", 0)
                annotations = result.get("annotations_processed", 0) 
                videos = result.get("downloaded_videos", 0)
                preprocessed = result.get("preprocessing", {}).get("processed_videos", 0)
                
                logger.info(f"  📊 {dataset_name}:")
                logger.info(f"    • {samples} samples total")
                logger.info(f"    • {annotations} with real annotations") 
                logger.info(f"    • {videos} videos downloaded")
                logger.info(f"    • {preprocessed} videos preprocessed")
        else:
            logger.warning("⚠️ No datasets downloaded")
    
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Test pipeline method (dry run)
    logger.info("\n🚀 Testing full pipeline (dry run)...")
    try:
        # This won't download videos but will show the structure
        pipeline_results = downloader.run_full_pipeline(
            datasets=["summe"], 
            max_videos_per_dataset=3,  # Very small for testing
            enable_preprocessing=False,  # Skip preprocessing for test
            cleanup_afterwards=False
        )
        
        if pipeline_results:
            logger.info(f"✅ Pipeline test successful!")
            logger.info(f"  📊 {pipeline_results['datasets_processed']} datasets")
            logger.info(f"  📝 {pipeline_results['total_samples']} total samples")
        
    except Exception as e:
        logger.error(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info("\n🎉 Dataset downloader testing complete!")

if __name__ == "__main__":
    test_dataset_downloader()