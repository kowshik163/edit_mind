"""
Auto Dataset Downloader - Automatically downloads and processes training datasets
"""

import os
import json
import requests
import subprocess
import logging
import zipfile
import tarfile
from pathlib import Path
from typing import Dict, List, Optional, Any
from urllib.parse import urlparse
from tqdm import tqdm
import pandas as pd

# Optional imports
try:
    import youtube_dl
    HAS_YOUTUBE_DL = True
except ImportError:
    HAS_YOUTUBE_DL = False

logger = logging.getLogger(__name__)


class DatasetAutoDownloader:
    """
    Automatically downloads and processes datasets for video editing training
    """
    
    def __init__(self, data_root: str = "data/datasets"):
        self.data_root = Path(data_root)
        self.data_root.mkdir(parents=True, exist_ok=True)
        
        # Dataset configurations with download URLs and processing info
        self.dataset_configs = {
            "webvid": {
                "name": "WebVid-10M",
                "urls": [
                    "https://huggingface.co/datasets/iejMac/CLIP-WebVid/resolve/main/results_2M_val.csv",
                    "https://huggingface.co/datasets/iejMac/CLIP-WebVid/resolve/main/results_2M_train.csv"
                ],
                "type": "csv",
                "video_col": "contentUrl",
                "text_col": "name",
                "sample_limit": 10000,  # Limit for demo
                "processor": self._process_webvid
            },
            "audioset": {
                "name": "AudioSet", 
                "urls": [
                    "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/balanced_train_segments.csv",
                    "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/eval_segments.csv"
                ],
                "type": "csv",
                "sample_limit": 5000,
                "processor": self._process_audioset
            },
            "activitynet": {
                "name": "ActivityNet",
                "urls": [
                    "http://ec2-52-25-205-214.us-west-2.compute.amazonaws.com/files/activity_net.v1-3.min.json"
                ],
                "type": "json",
                "sample_limit": 2000,
                "processor": self._process_activitynet
            },
            "tvsum": {
                "name": "TVSum",
                "urls": [
                    "https://github.com/yalesong/tvsum/raw/master/data/ydata-tvsum50-v1_1.tgz"
                ],
                "type": "tgz",
                "sample_limit": 50,
                "processor": self._process_tvsum
            },
            "summe": {
                "name": "SumMe",
                "urls": [
                    "https://gyglim.github.io/me/vsum/SumMe.zip"
                ],
                "type": "zip", 
                "sample_limit": 25,
                "processor": self._process_summe
            }
        }
        
        # Download session for retries
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def download_all_datasets(self, datasets: Optional[List[str]] = None,
                            force_download: bool = False) -> Dict[str, Any]:
        """Download all specified datasets"""
        
        if datasets is None:
            datasets = list(self.dataset_configs.keys())
        
        logger.info(f"📥 Starting auto-download for {len(datasets)} datasets...")
        
        results = {}
        
        for dataset_name in datasets:
            if dataset_name not in self.dataset_configs:
                logger.warning(f"Unknown dataset: {dataset_name}")
                continue
            
            logger.info(f"🔄 Processing {dataset_name}...")
            
            try:
                result = self._download_dataset(dataset_name, force_download)
                if result:
                    results[dataset_name] = result
                    logger.info(f"✅ {dataset_name}: {result.get('samples', 0)} samples")
                else:
                    logger.error(f"❌ Failed to download {dataset_name}")
                    
            except Exception as e:
                logger.error(f"❌ Error downloading {dataset_name}: {e}")
                continue
        
        # Save download manifest
        self._save_download_manifest(results)
        
        logger.info(f"🎉 Dataset download complete: {len(results)}/{len(datasets)} successful")
        return results
    
    def _download_dataset(self, dataset_name: str, force_download: bool = False) -> Optional[Dict[str, Any]]:
        """Download and process a single dataset"""
        
        config = self.dataset_configs[dataset_name]
        dataset_dir = self.data_root / dataset_name
        
        # Check if already exists
        if not force_download and dataset_dir.exists():
            processed_file = dataset_dir / "processed.json"
            if processed_file.exists():
                logger.info(f"  📋 Using cached {dataset_name}")
                with open(processed_file, 'r') as f:
                    return json.load(f)
        
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Download raw files
        raw_files = []
        for i, url in enumerate(config["urls"]):
            try:
                raw_file = self._download_file(url, dataset_dir, f"raw_{i}")
                if raw_file:
                    raw_files.append(raw_file)
                    
            except Exception as e:
                logger.warning(f"  ⚠️ Failed to download {url}: {e}")
                continue
        
        if not raw_files:
            logger.error(f"  ❌ No files downloaded for {dataset_name}")
            return None
        
        # Process dataset using specific processor
        try:
            result = config["processor"](dataset_name, raw_files, dataset_dir, config)
            
            # Save processed results
            if result:
                processed_file = dataset_dir / "processed.json"
                with open(processed_file, 'w') as f:
                    json.dump(result, f, indent=2)
            
            return result
            
        except Exception as e:
            logger.error(f"  ❌ Failed to process {dataset_name}: {e}")
            return None
    
    def _download_file(self, url: str, target_dir: Path, filename_prefix: str = "download") -> Optional[str]:
        """Download a single file with progress bar"""
        
        try:
            # Get filename from URL
            parsed = urlparse(url)
            filename = Path(parsed.path).name
            if not filename:
                filename = f"{filename_prefix}.download"
            
            target_path = target_dir / filename
            
            logger.info(f"    📡 Downloading {url}")
            
            # Stream download with progress bar
            response = self.session.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(target_path, 'wb') as f, tqdm(
                desc=filename[:30],
                total=total_size,
                unit='B',
                unit_scale=True
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
            
            logger.info(f"    ✅ Downloaded: {target_path}")
            return str(target_path)
            
        except Exception as e:
            logger.error(f"    ❌ Download failed: {e}")
            return None
    
    def _process_webvid(self, name: str, files: List[str], output_dir: Path, config: Dict) -> Dict[str, Any]:
        """Process WebVid CSV files"""
        
        logger.info(f"    🔄 Processing WebVid data...")
        
        all_samples = []
        
        for file_path in files:
            try:
                df = pd.read_csv(file_path)
                
                # Extract video and text data
                for _, row in df.iterrows():
                    if len(all_samples) >= config["sample_limit"]:
                        break
                        
                    sample = {
                        "video_url": row.get(config["video_col"], ""),
                        "text": row.get(config["text_col"], ""),
                        "duration": row.get("duration", 10.0),
                        "width": row.get("width", 640),
                        "height": row.get("height", 480),
                        "source": "webvid"
                    }
                    all_samples.append(sample)
                
                if len(all_samples) >= config["sample_limit"]:
                    break
                    
            except Exception as e:
                logger.warning(f"    ⚠️ Failed to process {file_path}: {e}")
                continue
        
        # Save processed samples
        samples_file = output_dir / "samples.json"
        with open(samples_file, 'w') as f:
            json.dump(all_samples, f, indent=2)
        
        return {
            "dataset": name,
            "samples": len(all_samples),
            "files": len(files),
            "samples_file": str(samples_file)
        }
    
    def _process_audioset(self, name: str, files: List[str], output_dir: Path, config: Dict) -> Dict[str, Any]:
        """Process AudioSet CSV files"""
        
        logger.info(f"    🔄 Processing AudioSet data...")
        
        all_samples = []
        
        for file_path in files:
            try:
                df = pd.read_csv(file_path, skiprows=3)  # AudioSet has 3 header rows
                
                for _, row in df.iterrows():
                    if len(all_samples) >= config["sample_limit"]:
                        break
                        
                    # AudioSet format: YTID, start_seconds, end_seconds, positive_labels
                    sample = {
                        "youtube_id": row.iloc[0] if len(row) > 0 else "",
                        "start_time": float(row.iloc[1]) if len(row) > 1 else 0.0,
                        "end_time": float(row.iloc[2]) if len(row) > 2 else 10.0,
                        "labels": str(row.iloc[3]) if len(row) > 3 else "",
                        "source": "audioset"
                    }
                    all_samples.append(sample)
                
                if len(all_samples) >= config["sample_limit"]:
                    break
                    
            except Exception as e:
                logger.warning(f"    ⚠️ Failed to process {file_path}: {e}")
                continue
        
        # Save processed samples
        samples_file = output_dir / "samples.json"
        with open(samples_file, 'w') as f:
            json.dump(all_samples, f, indent=2)
        
        return {
            "dataset": name,
            "samples": len(all_samples),
            "files": len(files),
            "samples_file": str(samples_file)
        }
    
    def _process_activitynet(self, name: str, files: List[str], output_dir: Path, config: Dict) -> Dict[str, Any]:
        """Process ActivityNet JSON files"""
        
        logger.info(f"    🔄 Processing ActivityNet data...")
        
        all_samples = []
        
        for file_path in files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # ActivityNet has 'database' with video info
                database = data.get('database', {})
                
                for video_id, video_info in database.items():
                    if len(all_samples) >= config["sample_limit"]:
                        break
                    
                    sample = {
                        "video_id": video_id,
                        "url": video_info.get('url', ''),
                        "duration": video_info.get('duration', 0),
                        "subset": video_info.get('subset', 'training'),
                        "annotations": video_info.get('annotations', []),
                        "source": "activitynet"
                    }
                    all_samples.append(sample)
                
                if len(all_samples) >= config["sample_limit"]:
                    break
                    
            except Exception as e:
                logger.warning(f"    ⚠️ Failed to process {file_path}: {e}")
                continue
        
        # Save processed samples
        samples_file = output_dir / "samples.json"
        with open(samples_file, 'w') as f:
            json.dump(all_samples, f, indent=2)
        
        return {
            "dataset": name,
            "samples": len(all_samples),
            "files": len(files),
            "samples_file": str(samples_file)
        }
    
    def _process_tvsum(self, name: str, files: List[str], output_dir: Path, config: Dict) -> Dict[str, Any]:
        """Process TVSum tgz files"""
        
        logger.info(f"    🔄 Processing TVSum data...")
        
        all_samples = []
        
        for file_path in files:
            try:
                # Extract tgz file
                with tarfile.open(file_path, 'r:gz') as tar:
                    tar.extractall(path=output_dir / "extracted")
                
                # Look for data files in extracted content
                extracted_dir = output_dir / "extracted"
                data_files = list(extracted_dir.rglob("*.mat")) + list(extracted_dir.rglob("*.json"))
                
                # Create sample entries (TVSum has 50 videos)
                for i in range(min(50, config["sample_limit"])):
                    sample = {
                        "video_id": f"tvsum_{i:03d}",
                        "dataset": "tvsum",
                        "type": "summarization",
                        "source": "tvsum"
                    }
                    all_samples.append(sample)
                    
            except Exception as e:
                logger.warning(f"    ⚠️ Failed to process {file_path}: {e}")
                continue
        
        # Save processed samples
        samples_file = output_dir / "samples.json"
        with open(samples_file, 'w') as f:
            json.dump(all_samples, f, indent=2)
        
        return {
            "dataset": name,
            "samples": len(all_samples),
            "files": len(files),
            "samples_file": str(samples_file)
        }
    
    def _process_summe(self, name: str, files: List[str], output_dir: Path, config: Dict) -> Dict[str, Any]:
        """Process SumMe zip files"""
        
        logger.info(f"    🔄 Processing SumMe data...")
        
        all_samples = []
        
        for file_path in files:
            try:
                # Extract zip file
                with zipfile.ZipFile(file_path, 'r') as zip_file:
                    zip_file.extractall(path=output_dir / "extracted")
                
                # Create sample entries (SumMe has 25 videos)
                for i in range(min(25, config["sample_limit"])):
                    sample = {
                        "video_id": f"summe_{i:03d}",
                        "dataset": "summe",
                        "type": "summarization", 
                        "source": "summe"
                    }
                    all_samples.append(sample)
                    
            except Exception as e:
                logger.warning(f"    ⚠️ Failed to process {file_path}: {e}")
                continue
        
        # Save processed samples
        samples_file = output_dir / "samples.json"
        with open(samples_file, 'w') as f:
            json.dump(all_samples, f, indent=2)
        
        return {
            "dataset": name,
            "samples": len(all_samples),
            "files": len(files),
            "samples_file": str(samples_file)
        }
    
    def _save_download_manifest(self, results: Dict[str, Any]):
        """Save download manifest"""
        
        manifest = {
            "downloaded_datasets": results,
            "data_root": str(self.data_root),
            "total_samples": sum(r.get("samples", 0) for r in results.values()),
            "datasets": list(results.keys())
        }
        
        manifest_file = self.data_root / "download_manifest.json"
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"📋 Saved download manifest: {manifest_file}")
    
    def get_dataset_info(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a downloaded dataset"""
        
        dataset_dir = self.data_root / dataset_name
        processed_file = dataset_dir / "processed.json"
        
        if processed_file.exists():
            with open(processed_file, 'r') as f:
                return json.load(f)
        
        return None
    
    def cleanup_raw_files(self):
        """Clean up raw download files to save space"""
        
        logger.info("🧹 Cleaning up raw download files...")
        
        for dataset_dir in self.data_root.iterdir():
            if not dataset_dir.is_dir():
                continue
                
            # Remove raw files, keep processed ones
            for file_path in dataset_dir.glob("raw_*"):
                try:
                    file_path.unlink()
                    logger.info(f"🗑️ Removed: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to remove {file_path}: {e}")


def auto_download_datasets(datasets: Optional[List[str]] = None,
                         data_root: str = "data/datasets",
                         force_download: bool = False) -> Dict[str, Any]:
    """
    Convenience function to auto-download datasets
    """
    downloader = DatasetAutoDownloader(data_root)
    return downloader.download_all_datasets(datasets, force_download)


if __name__ == "__main__":
    # CLI usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Auto-download video editing datasets")
    parser.add_argument("--data-dir", default="data/datasets", help="Data root directory")
    parser.add_argument("--datasets", nargs="+", help="Specific datasets to download")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    parser.add_argument("--cleanup", action="store_true", help="Cleanup raw files")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    downloader = DatasetAutoDownloader(args.data_dir)
    
    if args.cleanup:
        downloader.cleanup_raw_files()
    else:
        results = downloader.download_all_datasets(args.datasets, args.force)
        
        print(f"\n🎉 Download complete!")
        print(f"📦 Downloaded {len(results)} datasets")
        print(f"📊 Total samples: {sum(r.get('samples', 0) for r in results.values())}")
        print(f"💾 Data location: {args.data_dir}")
