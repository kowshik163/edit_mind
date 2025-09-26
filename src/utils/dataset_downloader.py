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


class DatasetDownloader:
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
                "processor": lambda *args, **kwargs: self._process_webvid(*args, **kwargs)
            },
            "howto100m": {
                "name": "HowTo100M",
                "urls": [
                    "https://www.di.ens.fr/willow/research/howto100m/HowTo100M_v1.csv"
                ],
                "type": "csv",
                "video_col": "video_id",
                "text_col": "text",
                "sample_limit": 50000,  # Large scale dataset
                "processor": lambda *args, **kwargs: self._process_howto100m(*args, **kwargs)
            },
            "audioset": {
                "name": "AudioSet", 
                "urls": [
                    "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/balanced_train_segments.csv",
                    "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/eval_segments.csv",
                    "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/unbalanced_train_segments.csv"
                ],
                "type": "csv",
                "sample_limit": 20000,  # Increased for better training
                "processor": lambda *args, **kwargs: self._process_audioset(*args, **kwargs)
            },
            "activitynet": {
                "name": "ActivityNet",
                "urls": [
                    "http://ec2-52-25-205-214.us-west-2.compute.amazonaws.com/files/activity_net.v1-3.min.json"
                ],
                "type": "json",
                "sample_limit": 5000,  # Increased
                "processor": lambda *args, **kwargs: self._process_activitynet(*args, **kwargs)
            },
            "ave_dataset": {
                "name": "Anatomy of Video Editing (AVE)",
                "urls": [
                    "https://github.com/jponttuset/shot-detection-papers/raw/master/data/AVE_dataset.json"
                ],
                "type": "json", 
                "sample_limit": 1000,
                "processor": lambda *args, **kwargs: self._process_ave_dataset(*args, **kwargs)
            },
            "v3c1": {
                "name": "V3C1 Dataset",
                "urls": [
                    "https://www.robots.ox.ac.uk/~vgg/data/v3c1/v3c1_meta.csv"
                ],
                "type": "csv",
                "sample_limit": 15000,
                "processor": lambda *args, **kwargs: self._process_v3c1(*args, **kwargs)
            },
            "reddit_editors": {
                "name": "Reddit r/editors Posts",
                "urls": [
                    "https://api.pushshift.io/reddit/search/submission/?subreddit=editors&size=1000",
                    "https://api.pushshift.io/reddit/search/submission/?subreddit=videoediting&size=1000",
                    "https://api.pushshift.io/reddit/search/submission/?subreddit=editing&size=1000"
                ],
                "type": "reddit_api",
                "sample_limit": 3000,
                "processor": lambda *args, **kwargs: self._process_reddit_editors(*args, **kwargs)
            },
            "youtube_tutorials": {
                "name": "YouTube Editing Tutorials",
                "urls": [],  # Will be populated dynamically
                "type": "youtube_channels",
                "channels": [
                    "UC8YmHryGLCkcE3KJop1d5Gg",  # Premiere Gal
                    "UC7O6CntQoAI-wYyJxYiqNUg",  # Justin Odisho  
                    "UCmXmlB4-HJytD7wek0Uo97A",  # Peter McKinnon
                    "UCQVD_BLufkoKQnWIPaUOagA"   # Matti Haapoja
                ],
                "sample_limit": 2000,
                "processor": lambda *args, **kwargs: self._process_youtube_tutorials(*args, **kwargs)
            },
            "kaggle_video_editing": {
                "name": "Kaggle Video Editing Datasets",
                "urls": [
                    "https://www.kaggle.com/datasets/gowrishankarp/camera-movements-and-angles-dataset",
                    "https://www.kaggle.com/datasets/kmader/film-grain-database"
                ],
                "type": "kaggle",
                "sample_limit": 5000,
                "processor": lambda *args, **kwargs: self._process_kaggle_datasets(*args, **kwargs)
            },
            "video_effects_scripts": {
                "name": "Video Effects Code Dataset",
                "urls": [
                    "https://github.com/search?q=ffmpeg+effects+language%3APython&type=code",
                    "https://github.com/search?q=opencv+video+effects+language%3APython&type=code"
                ],
                "type": "code_repository", 
                "sample_limit": 10000,
                "processor": lambda *args, **kwargs: self._process_video_effects_code(*args, **kwargs)
            },
            "professional_editing": {
                "name": "Professional Editing Patterns",
                "urls": [
                    "https://www.editingmentor.com/data/patterns.json",
                    "https://filmschoolonline.com/editing-techniques.json"
                ],
                "type": "json",
                "sample_limit": 1000, 
                "processor": lambda *args, **kwargs: self._process_professional_editing(*args, **kwargs)
            },
            "tvsum": {
                "name": "TVSum",
                "urls": [
                    "https://github.com/yalesong/tvsum/raw/master/data/ydata-tvsum50-v1_1.tgz"
                ],
                "type": "tgz",
                "sample_limit": 500,
                "processor": lambda *args, **kwargs: self._process_tvsum(*args, **kwargs)
            },
            "summe": {
                "name": "SumMe",
                "urls": [
                    "https://gyglim.github.io/me/vsum/SumMe.zip"
                ],
                "type": "zip", 
                "sample_limit": 500,
                "processor": lambda *args, **kwargs: self._process_summe(*args, **kwargs)
            }
        }
        
        # Download session for retries
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Initialize enhanced processors after all configs are set
        self._init_enhanced_processors()
    
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
                    videos_info = ""
                    if result.get("downloaded_videos", 0) > 0:
                        videos_info = f", {result['downloaded_videos']} videos downloaded"
                        if result.get("preprocessing", {}).get("processed_videos", 0) > 0:
                            videos_info += f", {result['preprocessing']['processed_videos']} preprocessed"
                    
                    logger.info(f"✅ {dataset_name}: {result.get('samples', 0)} samples{videos_info}")
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
        """Process TVSum tgz files and extract real annotations"""
        
        logger.info(f"    🔄 Processing TVSum data with real annotations...")
        
        all_samples = []
        
        for file_path in files:
            try:
                # Extract tgz file
                extracted_dir = output_dir / "extracted"
                extracted_dir.mkdir(parents=True, exist_ok=True)
                
                with tarfile.open(file_path, 'r:gz') as tar:
                    tar.extractall(path=extracted_dir)
                
                # Look for data files in extracted content
                mat_files = list(extracted_dir.rglob("*.mat"))
                json_files = list(extracted_dir.rglob("*.json"))
                
                # Process .mat files (original TVSum format)
                if mat_files:
                    all_samples.extend(self._process_tvsum_mat_files(mat_files, config))
                
                # Process .json files (if any converted format exists)
                if json_files:
                    all_samples.extend(self._process_tvsum_json_files(json_files, config))
                
                # If no annotation files found, create basic metadata
                if not mat_files and not json_files:
                    logger.warning(f"    ⚠️ No annotation files found in {file_path}, creating basic entries")
                    for i in range(min(50, config.get("sample_limit", 50))):
                        sample = {
                            "video_id": f"tvsum_{i:03d}",
                            "dataset": "tvsum",
                            "type": "summarization",
                            "source": "tvsum",
                            "annotations_available": False
                        }
                        all_samples.append(sample)
                        
            except Exception as e:
                logger.warning(f"    ⚠️ Failed to process {file_path}: {e}")
                continue
        
        # Save processed samples
        samples_file = output_dir / "samples.json"
        with open(samples_file, 'w') as f:
            json.dump(all_samples, f, indent=2)
        
        logger.info(f"    ✅ Processed {len(all_samples)} TVSum samples")
        
        # Try to download YouTube videos if video IDs are available
        downloaded_videos = []
        if all_samples:
            video_ids = [sample.get('video_id') for sample in all_samples if sample.get('video_id')]
            if video_ids:
                logger.info(f"📺 Found {len(video_ids)} video IDs, attempting to download...")
                videos_dir = output_dir / "videos"
                downloaded_videos = self.download_youtube_videos(video_ids[:20], videos_dir, max_videos=20)
        
        # Preprocess downloaded videos
        preprocessing_results = {}
        if downloaded_videos:
            logger.info("🔄 Preprocessing downloaded videos...")
            # Use the actual dataset name instead of hardcoded string
            preprocessing_results = self.preprocess_dataset_videos(name.lower() if name else "tvsum", max_videos=10)
        
        return {
            "dataset": name,
            "samples": len(all_samples),
            "files": len(files),
            "samples_file": str(samples_file),
            "annotations_processed": len([s for s in all_samples if s.get("annotations_available", False)]),
            "downloaded_videos": len(downloaded_videos),
            "video_data": downloaded_videos[:3] if downloaded_videos else [],
            "preprocessing": preprocessing_results
        }
    
    def _process_tvsum_mat_files(self, mat_files: List[Path], config: Dict) -> List[Dict[str, Any]]:
        """Process TVSum .mat annotation files"""
        samples = []
        
        try:
            # Try to import scipy for .mat file reading
            from scipy.io import loadmat
        except ImportError:
            logger.warning("    ⚠️ scipy not available, cannot read .mat files")
            return []
        
        for mat_file in mat_files:
            try:
                # Load .mat file
                mat_data = loadmat(str(mat_file))
                
                # TVSum .mat files typically contain:
                # - video: struct with video metadata
                # - user_anno: user annotations (importance scores)
                # - gt_score: ground truth importance scores
                
                if 'video' in mat_data:
                    video_data = mat_data['video'][0, 0]  # Extract struct
                    
                    # Extract basic video metadata
                    video_id = str(video_data['video_name'][0]) if 'video_name' in video_data.dtype.names else mat_file.stem
                    duration = float(video_data['length'][0, 0]) if 'length' in video_data.dtype.names else None
                    fps = float(video_data['fps'][0, 0]) if 'fps' in video_data.dtype.names else None
                    
                    # Extract importance scores if available
                    importance_scores = None
                    if 'gt_score' in mat_data:
                        importance_scores = mat_data['gt_score'].flatten().tolist()
                    
                    # Extract user annotations if available
                    user_annotations = None
                    if 'user_anno' in mat_data:
                        user_anno = mat_data['user_anno']
                        if user_anno.size > 0:
                            # Convert user annotations to list format
                            user_annotations = []
                            for i in range(user_anno.shape[0]):
                                user_annotations.append(user_anno[i].flatten().tolist())
                    
                    sample = {
                        "video_id": video_id,
                        "dataset": "tvsum",
                        "type": "summarization",
                        "source": "tvsum",
                        "annotations_available": True,
                        "duration": duration,
                        "fps": fps,
                        "importance_scores": importance_scores,
                        "user_annotations": user_annotations,
                        "annotation_file": str(mat_file)
                    }
                    
                    samples.append(sample)
                    
                    if len(samples) >= config.get("sample_limit", 50):
                        break
                        
            except Exception as e:
                logger.warning(f"    ⚠️ Failed to process {mat_file}: {e}")
                continue
        
        return samples
    
    def _process_tvsum_json_files(self, json_files: List[Path], config: Dict) -> List[Dict[str, Any]]:
        """Process TVSum JSON annotation files (if converted format available)"""
        samples = []
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Handle different JSON formats
                if isinstance(data, list):
                    # List of video annotations
                    for item in data:
                        if len(samples) >= config.get("sample_limit", 50):
                            break
                            
                        sample = {
                            "video_id": item.get("video_id", f"tvsum_{len(samples):03d}"),
                            "dataset": "tvsum",
                            "type": "summarization",
                            "source": "tvsum",
                            "annotations_available": True,
                            "duration": item.get("duration"),
                            "fps": item.get("fps"),
                            "importance_scores": item.get("importance_scores"),
                            "user_annotations": item.get("user_annotations"),
                            "annotation_file": str(json_file)
                        }
                        samples.append(sample)
                        
                elif isinstance(data, dict):
                    # Single video annotation
                    sample = {
                        "video_id": data.get("video_id", json_file.stem),
                        "dataset": "tvsum",
                        "type": "summarization",
                        "source": "tvsum", 
                        "annotations_available": True,
                        "duration": data.get("duration"),
                        "fps": data.get("fps"),
                        "importance_scores": data.get("importance_scores"),
                        "user_annotations": data.get("user_annotations"),
                        "annotation_file": str(json_file)
                    }
                    samples.append(sample)
                    
            except Exception as e:
                logger.warning(f"    ⚠️ Failed to process {json_file}: {e}")
                continue
        
        return samples
    
    def _process_summe(self, name: str, files: List[str], output_dir: Path, config: Dict) -> Dict[str, Any]:
        """Process SumMe zip files and extract real annotations"""
        
        logger.info(f"    🔄 Processing SumMe data with real annotations...")
        
        all_samples = []
        
        for file_path in files:
            try:
                # Extract zip file
                extracted_dir = output_dir / "extracted"
                extracted_dir.mkdir(parents=True, exist_ok=True)
                
                with zipfile.ZipFile(file_path, 'r') as zip_file:
                    zip_file.extractall(path=extracted_dir)
                
                # Look for SumMe annotation files
                mat_files = list(extracted_dir.rglob("*.mat"))
                json_files = list(extracted_dir.rglob("*.json"))
                txt_files = list(extracted_dir.rglob("*.txt"))
                
                # Process different file types
                if mat_files:
                    all_samples.extend(self._process_summe_mat_files(mat_files, config))
                
                if json_files:
                    all_samples.extend(self._process_summe_json_files(json_files, config))
                
                if txt_files:
                    all_samples.extend(self._process_summe_txt_files(txt_files, config))
                
                # If no annotation files found, create basic metadata
                if not mat_files and not json_files and not txt_files:
                    logger.warning(f"    ⚠️ No annotation files found in {file_path}, creating basic entries")
                    for i in range(min(25, config.get("sample_limit", 25))):
                        sample = {
                            "video_id": f"summe_{i:03d}",
                            "dataset": "summe",
                            "type": "summarization", 
                            "source": "summe",
                            "annotations_available": False
                        }
                        all_samples.append(sample)
                        
            except Exception as e:
                logger.warning(f"    ⚠️ Failed to process {file_path}: {e}")
                continue
        
        # Save processed samples
        samples_file = output_dir / "samples.json"
        with open(samples_file, 'w') as f:
            json.dump(all_samples, f, indent=2)
        
        logger.info(f"    ✅ Processed {len(all_samples)} SumMe samples")
        
        # Try to download YouTube videos if video IDs are available
        downloaded_videos = []
        if all_samples:
            video_ids = [sample.get('video_id') for sample in all_samples if sample.get('video_id')]
            if video_ids:
                logger.info(f"📺 Found {len(video_ids)} video IDs, attempting to download...")
                videos_dir = output_dir / "videos"
                downloaded_videos = self.download_youtube_videos(video_ids[:15], videos_dir, max_videos=15)
        
        # Preprocess downloaded videos
        preprocessing_results = {}
        if downloaded_videos:
            logger.info("🔄 Preprocessing downloaded videos...")
            preprocessing_results = self.preprocess_dataset_videos(name.lower() if name else "summe", max_videos=8)
        
        return {
            "dataset": name,
            "samples": len(all_samples),
            "files": len(files),
            "samples_file": str(samples_file),
            "annotations_processed": len([s for s in all_samples if s.get("annotations_available", False)]),
            "downloaded_videos": len(downloaded_videos),
            "video_data": downloaded_videos[:3] if downloaded_videos else [],
            "preprocessing": preprocessing_results
        }
    
    def _process_summe_mat_files(self, mat_files: List[Path], config: Dict) -> List[Dict[str, Any]]:
        """Process SumMe .mat annotation files"""
        samples = []
        
        try:
            from scipy.io import loadmat
        except ImportError:
            logger.warning("    ⚠️ scipy not available, cannot read .mat files")
            return []
        
        for mat_file in mat_files:
            try:
                # Load .mat file
                mat_data = loadmat(str(mat_file))
                
                # SumMe .mat files typically contain:
                # - video_name: name of the video
                # - user_score: user importance scores
                # - gt_score: ground truth summary
                
                video_name = None
                if 'video_name' in mat_data:
                    video_name = str(mat_data['video_name'][0])
                else:
                    video_name = mat_file.stem
                
                # Extract user scores (multiple annotators)
                user_scores = None
                if 'user_score' in mat_data:
                    user_scores = mat_data['user_score']
                    if user_scores.size > 0:
                        # Convert to list of lists (one per annotator)
                        user_scores = user_scores.tolist()
                
                # Extract ground truth scores
                gt_scores = None
                if 'gt_score' in mat_data:
                    gt_scores = mat_data['gt_score'].flatten().tolist()
                
                # Extract other metadata if available
                duration = None
                fps = None
                if 'video_duration' in mat_data:
                    duration = float(mat_data['video_duration'][0, 0])
                if 'fps' in mat_data:
                    fps = float(mat_data['fps'][0, 0])
                
                sample = {
                    "video_id": video_name,
                    "dataset": "summe",
                    "type": "summarization",
                    "source": "summe",
                    "annotations_available": True,
                    "duration": duration,
                    "fps": fps,
                    "user_scores": user_scores,
                    "gt_scores": gt_scores,
                    "annotation_file": str(mat_file)
                }
                
                samples.append(sample)
                
                if len(samples) >= config.get("sample_limit", 25):
                    break
                    
            except Exception as e:
                logger.warning(f"    ⚠️ Failed to process {mat_file}: {e}")
                continue
        
        return samples
    
    def _process_summe_json_files(self, json_files: List[Path], config: Dict) -> List[Dict[str, Any]]:
        """Process SumMe JSON annotation files"""
        samples = []
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    for item in data:
                        if len(samples) >= config.get("sample_limit", 25):
                            break
                            
                        sample = {
                            "video_id": item.get("video_id", f"summe_{len(samples):03d}"),
                            "dataset": "summe",
                            "type": "summarization",
                            "source": "summe",
                            "annotations_available": True,
                            "duration": item.get("duration"),
                            "fps": item.get("fps"),
                            "user_scores": item.get("user_scores"),
                            "gt_scores": item.get("gt_scores"),
                            "annotation_file": str(json_file)
                        }
                        samples.append(sample)
                        
                elif isinstance(data, dict):
                    sample = {
                        "video_id": data.get("video_id", json_file.stem),
                        "dataset": "summe",
                        "type": "summarization",
                        "source": "summe",
                        "annotations_available": True,
                        "duration": data.get("duration"),
                        "fps": data.get("fps"),
                        "user_scores": data.get("user_scores"),
                        "gt_scores": data.get("gt_scores"),
                        "annotation_file": str(json_file)
                    }
                    samples.append(sample)
                    
            except Exception as e:
                logger.warning(f"    ⚠️ Failed to process {json_file}: {e}")
                continue
        
        return samples
    
    def _process_summe_txt_files(self, txt_files: List[Path], config: Dict) -> List[Dict[str, Any]]:
        """Process SumMe text annotation files"""
        samples = []
        
        for txt_file in txt_files:
            try:
                with open(txt_file, 'r') as f:
                    lines = f.readlines()
                
                # Parse text format - varies by SumMe version
                # Common format: video_name, scores, or metadata
                
                for line in lines:
                    if len(samples) >= config.get("sample_limit", 25):
                        break
                    
                    line = line.strip()
                    if not line or line.startswith('#'):  # Skip comments
                        continue
                    
                    # Try to parse as comma-separated values
                    parts = [p.strip() for p in line.split(',')]
                    
                    if len(parts) >= 1:
                        video_id = parts[0]
                        
                        # Extract numeric scores if available
                        scores = []
                        for part in parts[1:]:
                            try:
                                scores.append(float(part))
                            except ValueError:
                                continue
                        
                        sample = {
                            "video_id": video_id,
                            "dataset": "summe", 
                            "type": "summarization",
                            "source": "summe",
                            "annotations_available": True,
                            "scores": scores if scores else None,
                            "annotation_file": str(txt_file)
                        }
                        samples.append(sample)
                        
            except Exception as e:
                logger.warning(f"    ⚠️ Failed to process {txt_file}: {e}")
                continue
        
        return samples
    
    def download_youtube_videos(self, video_ids: List[str], output_dir: Path, max_videos: int = 100) -> List[Dict[str, Any]]:
        """Download videos from YouTube using video IDs"""
        
        if not HAS_YOUTUBE_DL:
            try:
                import yt_dlp as youtube_dl
                logger.info("Using yt-dlp instead of youtube-dl")
            except ImportError:
                logger.error("Neither youtube-dl nor yt-dlp available. Install with: pip install yt-dlp")
                return []
        
        downloaded_videos = []
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure youtube-dl options
        ydl_opts = {
            'outtmpl': str(output_dir / '%(id)s.%(ext)s'),
            'format': 'best[height<=720]',  # Limit to 720p max
            'extract_flat': False,
            'writeinfojson': True,  # Save metadata
            'writethumbnail': True,
            'ignoreerrors': True,
        }
        
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            for i, video_id in enumerate(video_ids[:max_videos]):
                try:
                    logger.info(f"    📥 Downloading video {i+1}/{min(len(video_ids), max_videos)}: {video_id}")
                    
                    # Construct YouTube URL
                    url = f"https://www.youtube.com/watch?v={video_id}"
                    
                    # Extract video info first
                    info = ydl.extract_info(url, download=False)
                    
                    # Check if video is available and not too long
                    duration = info.get('duration', 0)
                    if duration > 600:  # Skip videos longer than 10 minutes
                        logger.warning(f"    ⏩ Skipping long video {video_id} ({duration}s)")
                        continue
                    
                    # Download the video
                    ydl.download([url])
                    
                    # Create metadata entry
                    video_metadata = {
                        "video_id": video_id,
                        "title": info.get('title', ''),
                        "duration": duration,
                        "view_count": info.get('view_count', 0),
                        "upload_date": info.get('upload_date', ''),
                        "uploader": info.get('uploader', ''),
                        "description": info.get('description', ''),
                        "categories": info.get('categories', []),
                        "tags": info.get('tags', []),
                        "url": url,
                        "local_path": str(output_dir / f"{video_id}.{info.get('ext', 'mp4')}"),
                        "thumbnail_path": str(output_dir / f"{video_id}.{info.get('thumbnail', 'jpg')}"),
                        "info_path": str(output_dir / f"{video_id}.info.json")
                    }
                    
                    downloaded_videos.append(video_metadata)
                    
                except Exception as e:
                    logger.warning(f"    ⚠️ Failed to download {video_id}: {e}")
                    continue
        
        logger.info(f"    ✅ Downloaded {len(downloaded_videos)} YouTube videos")
        return downloaded_videos
    
    def extract_video_features(self, video_path: str, output_dir: Path) -> Dict[str, Any]:
        """Extract features from video files (frames, audio, metadata)"""
        
        try:
            import cv2
        except ImportError:
            logger.warning("OpenCV not available, skipping video processing")
            return {}
        
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        features = {
            "video_path": str(video_path),
            "frames_extracted": 0,
            "audio_extracted": False,
            "metadata": {}
        }
        
        try:
            # Open video file
            cap = cv2.VideoCapture(str(video_path))
            
            # Get video metadata
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            features["metadata"] = {
                "fps": fps,
                "frame_count": frame_count,
                "width": width,
                "height": height,
                "duration": duration
            }
            
            # Extract frames at regular intervals
            frames_dir = output_dir / "frames"
            frames_dir.mkdir(exist_ok=True)
            
            # Sample frames (e.g., every 30 frames or every second)
            frame_interval = max(1, int(fps))  # Sample every second
            frame_idx = 0
            extracted_frames = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % frame_interval == 0:
                    # Save frame
                    frame_filename = frames_dir / f"frame_{extracted_frames:06d}.jpg"
                    cv2.imwrite(str(frame_filename), frame)
                    extracted_frames += 1
                    
                    # Limit number of extracted frames
                    if extracted_frames >= 100:  # Max 100 frames per video
                        break
                
                frame_idx += 1
            
            cap.release()
            features["frames_extracted"] = extracted_frames
            
            # Extract audio using ffmpeg if available
            audio_path = output_dir / f"{video_path.stem}.wav"
            try:
                import subprocess
                ffmpeg_cmd = [
                    'ffmpeg', '-i', str(video_path),
                    '-vn',  # No video
                    '-acodec', 'pcm_s16le',  # PCM 16-bit
                    '-ar', '16000',  # 16kHz sample rate
                    '-ac', '1',  # Mono
                    '-y',  # Overwrite
                    str(audio_path)
                ]
                
                result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    features["audio_extracted"] = True
                    features["audio_path"] = str(audio_path)
                else:
                    logger.debug(f"FFmpeg audio extraction failed: {result.stderr}")
                    
            except FileNotFoundError:
                logger.debug("FFmpeg not found, skipping audio extraction")
            except Exception as e:
                logger.debug(f"Audio extraction failed: {e}")
            
        except Exception as e:
            logger.warning(f"Video feature extraction failed for {video_path}: {e}")
        
        return features
    
    def preprocess_dataset_videos(self, dataset_name: str, max_videos: int = 50) -> Dict[str, Any]:
        """Preprocess videos for a specific dataset (extract frames, audio, etc.)"""
        
        dataset_dir = self.data_root / dataset_name
        if not dataset_dir.exists():
            logger.warning(f"Dataset directory not found: {dataset_dir}")
            return {}
        
        # Look for downloaded videos
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(dataset_dir.rglob(f'*{ext}'))
        
        if not video_files:
            logger.warning(f"No video files found in {dataset_dir}")
            return {}
        
        # Create preprocessing output directory
        preprocess_dir = dataset_dir / "preprocessed"
        preprocess_dir.mkdir(exist_ok=True)
        
        processed_videos = []
        
        for i, video_file in enumerate(video_files[:max_videos]):
            try:
                logger.info(f"    🔄 Processing video {i+1}/{min(len(video_files), max_videos)}: {video_file.name}")
                
                # Create output directory for this video
                video_output_dir = preprocess_dir / video_file.stem
                
                # Extract features
                features = self.extract_video_features(video_file, video_output_dir)
                features["video_name"] = video_file.name
                features["video_id"] = video_file.stem
                
                processed_videos.append(features)
                
            except Exception as e:
                logger.warning(f"    ⚠️ Failed to process {video_file}: {e}")
                continue
        
        # Save preprocessing results
        results = {
            "dataset": dataset_name,
            "processed_videos": len(processed_videos),
            "videos": processed_videos,
            "preprocess_dir": str(preprocess_dir)
        }
        
        results_file = preprocess_dir / "preprocessing_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"    ✅ Preprocessed {len(processed_videos)} videos for {dataset_name}")
        
        return results
    
    def cleanup_raw_files(self, keep_preprocessed: bool = True):
        """Clean up raw downloaded files to save space"""
        
        logger.info("🧹 Cleaning up raw files...")
        
        total_size_freed = 0
        
        for dataset_dir in self.data_root.iterdir():
            if not dataset_dir.is_dir():
                continue
            
            # Look for large raw files
            raw_extensions = ['.zip', '.tar.gz', '.tgz', '.csv'] 
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
            
            # Remove raw archives after processing
            for ext in raw_extensions:
                for file_path in dataset_dir.rglob(f'*{ext}'):
                    if file_path.is_file():
                        size = file_path.stat().st_size
                        file_path.unlink()
                        total_size_freed += size
                        logger.info(f"    🗑️ Removed {file_path.name} ({size // (1024*1024)} MB)")
            
            # Optionally remove original videos if preprocessed versions exist
            if not keep_preprocessed:
                preprocessed_dir = dataset_dir / "preprocessed"
                if preprocessed_dir.exists():
                    for ext in video_extensions:
                        for video_file in dataset_dir.glob(f'*{ext}'):
                            if video_file.is_file():
                                size = video_file.stat().st_size
                                video_file.unlink()
                                total_size_freed += size
                                logger.info(f"    🗑️ Removed {video_file.name} ({size // (1024*1024)} MB)")
        
        logger.info(f"🎉 Cleanup complete! Freed {total_size_freed // (1024*1024)} MB")
    
    def run_full_pipeline(self, datasets: Optional[List[str]] = None, 
                         max_videos_per_dataset: int = 10,
                         enable_preprocessing: bool = True,
                         cleanup_afterwards: bool = False) -> Dict[str, Any]:
        """Run the complete dataset download, video processing, and preprocessing pipeline"""
        
        logger.info("🚀 Starting full dataset processing pipeline...")
        
        # Step 1: Download datasets with annotations
        logger.info("📋 Phase 1: Downloading datasets and parsing annotations...")
        results = self.download_all_datasets(datasets)
        
        if not results:
            logger.error("❌ No datasets downloaded successfully")
            return {}
        
        # Step 2: Additional video downloading if needed
        logger.info("📺 Phase 2: Additional video downloading (if needed)...")
        total_videos_downloaded = sum(r.get("downloaded_videos", 0) for r in results.values())
        
        # Step 3: Preprocessing downloaded videos
        if enable_preprocessing and total_videos_downloaded > 0:
            logger.info("🔄 Phase 3: Video preprocessing (frame extraction, audio)...")
            for dataset_name, result in results.items():
                if result.get("downloaded_videos", 0) > 0:
                    logger.info(f"  🔄 Processing {dataset_name} videos...")
                    preprocessing_result = self.preprocess_dataset_videos(
                        dataset_name, 
                        max_videos=max_videos_per_dataset
                    )
                    result["additional_preprocessing"] = preprocessing_result
        
        # Step 4: Cleanup if requested
        if cleanup_afterwards:
            logger.info("🧹 Phase 4: Cleaning up raw files...")
            self.cleanup_raw_files(keep_preprocessed=True)
        
        # Generate final summary
        total_samples = sum(r.get("samples", 0) for r in results.values())
        total_videos = sum(r.get("downloaded_videos", 0) for r in results.values())
        total_preprocessed = sum(
            r.get("preprocessing", {}).get("processed_videos", 0) 
            for r in results.values()
        )
        
        pipeline_summary = {
            "datasets_processed": len(results),
            "total_samples": total_samples,
            "total_videos_downloaded": total_videos,
            "total_videos_preprocessed": total_preprocessed,
            "results": results
        }
        
        logger.info("🎉 Full pipeline complete!")
        logger.info(f"  📊 {len(results)} datasets, {total_samples} samples")
        logger.info(f"  📺 {total_videos} videos downloaded, {total_preprocessed} preprocessed")
        
        return pipeline_summary
    
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
    
    def _init_enhanced_processors(self):
        """Initialize enhanced dataset processors via bridge"""
        try:
            from .enhanced_processor_bridge import EnhancedProcessorBridge
            bridge = EnhancedProcessorBridge(self)
            
            # Set enhanced processors
            self._process_ave_dataset = bridge.process_ave_dataset
            self._process_v3c1 = bridge.process_v3c1
            self._process_reddit_editors = bridge.process_reddit_editors
            self._process_youtube_tutorials = bridge.process_youtube_tutorials
            self._process_video_effects_code = bridge.process_video_effects_code
            self._process_kaggle_datasets = bridge.process_kaggle_datasets
            self._process_professional_editing = bridge.process_professional_editing
            
        except Exception as e:
            logger.error(f"Failed to initialize enhanced processors: {e}")
            # Set basic fallback processors
            self._setup_fallback_processors()
    
    def _setup_fallback_processors(self):
        """Setup basic fallback processors"""
        def basic_fallback(name: str, files: List[str], output_dir: Path, config: Dict) -> Dict[str, Any]:
            logger.warning(f"Using basic fallback for {name}")
            all_samples = [{"sample_id": f"{name}_fallback", "source": name, "fallback": True}]
            samples_file = output_dir / "samples.json"
            with open(samples_file, 'w') as f:
                json.dump(all_samples, f, indent=2)
            return {"dataset": name, "samples": len(all_samples), "files": len(files), "samples_file": str(samples_file), "fallback": True}
        
        # Assign fallback to all enhanced processors
        self._process_ave_dataset = basic_fallback
        self._process_v3c1 = basic_fallback
        self._process_reddit_editors = basic_fallback
        self._process_youtube_tutorials = basic_fallback
        self._process_video_effects_code = basic_fallback
        self._process_kaggle_datasets = basic_fallback
        self._process_professional_editing = basic_fallback


def auto_download_datasets(datasets: Optional[List[str]] = None,
                         data_root: str = "data/datasets",
                         force_download: bool = False) -> Dict[str, Any]:
    """
    Convenience function to auto-download datasets
    """
    downloader = DatasetDownloader(data_root)
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
    
    downloader = DatasetDownloader(args.data_dir)
    
    if args.cleanup:
        downloader.cleanup_raw_files()
    else:
        results = downloader.download_all_datasets(args.datasets, args.force)
        
        print(f"\n🎉 Download complete!")
        print(f"📦 Downloaded {len(results)} datasets")
        print(f"📊 Total samples: {sum(r.get('samples', 0) for r in results.values())}")
        print(f"💾 Data location: {args.data_dir}")
        
    def _process_howto100m(self, name: str, files: List[str], output_dir: Path, config: Dict) -> Dict[str, Any]:
        """Process HowTo100M dataset"""
        
        logger.info(f"    🔄 Processing HowTo100M data...")
        all_samples = []
        
        for file_path in files:
            try:
                df = pd.read_csv(file_path)
                
                for _, row in df.iterrows():
                    if len(all_samples) >= config["sample_limit"]:
                        break
                    
                    sample = {
                        "video_id": row.get("video_id", ""),
                        "text": row.get("text", ""),
                        "start": row.get("start", 0.0),
                        "end": row.get("end", 10.0),
                        "task_type": row.get("task_type", "general"),
                        "source": "howto100m"
                    }
                    all_samples.append(sample)
                
                if len(all_samples) >= config["sample_limit"]:
                    break
                    
            except Exception as e:
                logger.warning(f"    ⚠️ Failed to process {file_path}: {e}")
        
        samples_file = output_dir / "samples.json"
        with open(samples_file, 'w') as f:
            json.dump(all_samples, f, indent=2)
        
        return {
            "dataset": name,
            "samples": len(all_samples),
            "files": len(files),
            "samples_file": str(samples_file)
        }
