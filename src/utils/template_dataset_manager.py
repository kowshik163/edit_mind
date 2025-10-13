"""
Template Dataset Manager for Video Editing Training
Handles downloading and processing of free video templates from various sources
"""

import os
import json
import logging
import requests
import zipfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import xml.etree.ElementTree as ET
from urllib.parse import urljoin, urlparse
import hashlib
import time
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np
from bs4 import BeautifulSoup
import yt_dlp

logger = logging.getLogger(__name__)


class TemplateFormat(Enum):
    """Supported template formats"""
    AFTER_EFFECTS = "aep"
    PREMIERE_PRO = "prproj" 
    DAVINCI_RESOLVE = "drp"
    CAPCUT = "cct"
    MOTION_GRAPHICS = "mogrt"
    VIDEO_FILE = "mp4"
    PROJECT_FILE = "json"


@dataclass
class TemplateMetadata:
    """Template metadata structure"""
    id: str
    name: str
    category: str
    format: TemplateFormat
    duration: float
    fps: int
    resolution: Tuple[int, int]
    description: str
    tags: List[str]
    source: str
    download_url: str
    file_path: Optional[str] = None
    beat_markers: Optional[List[float]] = None
    transition_points: Optional[List[float]] = None
    text_layers: Optional[List[Dict]] = None
    audio_sync: Optional[Dict] = None


class TemplateDatasetManager:
    """
    Manages downloading and processing of video editing templates from free sources
    """
    
    def __init__(self, data_root: str = "data/templates"):
        self.data_root = Path(data_root)
        self.data_root.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different sources
        self.sources_dir = {
            'mixkit': self.data_root / 'mixkit',
            'videezy': self.data_root / 'videezy', 
            'capcut': self.data_root / 'capcut',
            'motion_array': self.data_root / 'motion_array',
            'canva': self.data_root / 'canva',
            'youtube_creators': self.data_root / 'youtube_creators',
            'stock_footage': self.data_root / 'stock_footage',
            'processed': self.data_root / 'processed',
            'metadata': self.data_root / 'metadata'
        }
        
        for dir_path in self.sources_dir.values():
            dir_path.mkdir(parents=True, exist_ok=True)
            
        self.templates_metadata = []
        self.download_stats = {
            'total_downloads': 0,
            'successful_downloads': 0,
            'failed_downloads': 0,
            'processing_errors': 0
        }
        
    def download_all_template_datasets(self) -> Dict[str, Any]:
        """
        Download templates from all available free sources
        """
        logger.info("Starting comprehensive template dataset download...")
        
        results = {}
        
        # 1. Download Mixkit templates
        results['mixkit'] = self._download_mixkit_templates()
        
        # 2. Download Videezy templates  
        results['videezy'] = self._download_videezy_templates()
        
        # 3. Download CapCut templates (via API/scraping)
        results['capcut'] = self._download_capcut_templates()
        
        # 4. Download Motion Array free section
        results['motion_array'] = self._download_motion_array_free()
        
        # 5. Download Canva video templates
        results['canva'] = self._download_canva_templates()
        
        # 6. Scrape YouTube creator template packs
        results['youtube_creators'] = self._download_youtube_creator_packs()
        
        # 7. Download stock footage for pairing
        results['stock_footage'] = self._download_stock_footage()
        
        # 8. Process and organize all templates
        results['processing'] = self._process_all_templates()
        
        logger.info(f"Template download complete. Results: {results}")
        return results
        
    def _download_mixkit_templates(self) -> Dict[str, Any]:
        """Download free templates from Mixkit"""
        logger.info("Downloading Mixkit templates...")
        
        mixkit_categories = [
            'video-templates/intros',
            'video-templates/transitions', 
            'video-templates/titles',
            'video-templates/lower-thirds',
            'video-templates/effects'
        ]
        
        downloaded = []
        
        for category in mixkit_categories:
            try:
                # Scrape Mixkit category pages
                url = f"https://mixkit.co/{category}/"
                response = requests.get(url, timeout=30)
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find template download links
                template_links = soup.find_all('a', class_='template-card')
                
                for link in template_links[:10]:  # Limit per category
                    template_url = urljoin(url, link.get('href'))
                    template_data = self._download_single_mixkit_template(template_url, category)
                    if template_data:
                        downloaded.append(template_data)
                        
            except Exception as e:
                logger.error(f"Error downloading Mixkit category {category}: {e}")
                
        return {
            'source': 'mixkit',
            'downloaded_count': len(downloaded),
            'templates': downloaded
        }
        
    def _download_single_mixkit_template(self, template_url: str, category: str) -> Optional[TemplateMetadata]:
        """Download a single Mixkit template"""
        try:
            response = requests.get(template_url, timeout=30)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract template metadata
            title = soup.find('h1', class_='template-title')
            title = title.text.strip() if title else "Unknown Template"
            
            # Find download button
            download_btn = soup.find('a', class_='download-btn')
            if not download_btn:
                return None
                
            download_url = download_btn.get('href')
            if not download_url:
                return None
                
            # Create template metadata
            template_id = hashlib.md5(template_url.encode()).hexdigest()[:12]
            
            metadata = TemplateMetadata(
                id=template_id,
                name=title,
                category=category.split('/')[-1],
                format=TemplateFormat.AFTER_EFFECTS,  # Most Mixkit templates are AE
                duration=0.0,  # Will be extracted later
                fps=30,
                resolution=(1920, 1080),
                description=f"Mixkit template from {category}",
                tags=['mixkit', category.split('/')[-1]],
                source='mixkit',
                download_url=download_url
            )
            
            # Download the template file
            file_path = self._download_template_file(download_url, self.sources_dir['mixkit'], template_id)
            if file_path:
                metadata.file_path = str(file_path)
                self.templates_metadata.append(metadata)
                return metadata
                
        except Exception as e:
            logger.error(f"Error downloading Mixkit template {template_url}: {e}")
            
        return None
        
    def _download_videezy_templates(self) -> Dict[str, Any]:
        """Download free templates from Videezy"""
        logger.info("Downloading Videezy templates...")
        
        # Videezy has free motion graphics and templates
        categories = [
            'motion-graphics/free',
            'video-templates/free', 
            'overlays/free'
        ]
        
        downloaded = []
        
        for category in categories:
            try:
                url = f"https://videezy.com/{category}/"
                response = requests.get(url, timeout=30)
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find free template links
                template_cards = soup.find_all('div', class_='video-card')
                
                for card in template_cards[:10]:  # Limit per category
                    link = card.find('a')
                    if link:
                        template_url = urljoin(url, link.get('href'))
                        template_data = self._download_single_videezy_template(template_url, category)
                        if template_data:
                            downloaded.append(template_data)
                            
            except Exception as e:
                logger.error(f"Error downloading Videezy category {category}: {e}")
                
        return {
            'source': 'videezy',
            'downloaded_count': len(downloaded),
            'templates': downloaded
        }
        
    def _download_single_videezy_template(self, template_url: str, category: str) -> Optional[TemplateMetadata]:
        """Download a single Videezy template"""
        try:
            response = requests.get(template_url, timeout=30)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            title = soup.find('h1', class_='video-title')
            title = title.text.strip() if title else "Unknown Template"
            
            # Look for free download button
            download_section = soup.find('div', class_='download-section')
            if not download_section:
                return None
                
            free_download = download_section.find('a', string=lambda text: text and 'free' in text.lower())
            if not free_download:
                return None
                
            download_url = free_download.get('href')
            
            template_id = hashlib.md5(template_url.encode()).hexdigest()[:12]
            
            metadata = TemplateMetadata(
                id=template_id,
                name=title,
                category=category.split('/')[-2],  # Remove 'free' suffix
                format=TemplateFormat.VIDEO_FILE,  # Most Videezy are video files
                duration=0.0,
                fps=30,
                resolution=(1920, 1080),
                description=f"Videezy template from {category}",
                tags=['videezy', category.split('/')[-2]],
                source='videezy',
                download_url=download_url
            )
            
            file_path = self._download_template_file(download_url, self.sources_dir['videezy'], template_id)
            if file_path:
                metadata.file_path = str(file_path)
                self.templates_metadata.append(metadata)
                return metadata
                
        except Exception as e:
            logger.error(f"Error downloading Videezy template {template_url}: {e}")
            
        return None
        
    def _download_capcut_templates(self) -> Dict[str, Any]:
        """Download CapCut templates (phonk beats, viral edits, etc.)"""
        logger.info("Downloading CapCut templates...")
        
        # CapCut templates are usually accessed via their API or app
        # For now, we'll create placeholder structure for when API access is available
        
        capcut_categories = [
            'trending', 'phonk', 'beat_sync', 'transitions', 'effects'
        ]
        
        downloaded = []
        
        # Placeholder implementation - would need CapCut API access
        for category in capcut_categories:
            try:
                # This would be replaced with actual CapCut API calls
                placeholder_templates = self._create_capcut_placeholder_templates(category)
                downloaded.extend(placeholder_templates)
                
            except Exception as e:
                logger.error(f"Error with CapCut category {category}: {e}")
                
        return {
            'source': 'capcut',
            'downloaded_count': len(downloaded),
            'templates': downloaded,
            'note': 'Placeholder implementation - requires CapCut API access'
        }
        
    def _create_capcut_placeholder_templates(self, category: str) -> List[TemplateMetadata]:
        """Create placeholder CapCut template metadata"""
        templates = []
        
        # Create a few placeholder templates for each category
        for i in range(3):
            template_id = f"capcut_{category}_{i:03d}"
            
            metadata = TemplateMetadata(
                id=template_id,
                name=f"CapCut {category.title()} Template {i+1}",
                category=category,
                format=TemplateFormat.CAPCUT,
                duration=15.0,  # Typical short-form duration
                fps=30,
                resolution=(1080, 1920),  # Vertical for TikTok/Instagram
                description=f"CapCut {category} template with beat synchronization",
                tags=['capcut', category, 'short_form', 'vertical'],
                source='capcut',
                download_url=f"capcut://template/{template_id}",
                beat_markers=[0.5, 1.0, 1.5, 2.0, 2.5] if category == 'phonk' else None
            )
            
            templates.append(metadata)
            
        return templates
        
    def _download_motion_array_free(self) -> Dict[str, Any]:
        """Download free templates from Motion Array"""
        logger.info("Downloading Motion Array free templates...")
        
        downloaded = []
        
        try:
            url = "https://motionarray.com/browse/free/"
            response = requests.get(url, timeout=30)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find free template listings
            template_cards = soup.find_all('div', class_='browse-item')
            
            for card in template_cards[:15]:  # Limit downloads
                link = card.find('a')
                if link:
                    template_url = urljoin(url, link.get('href'))
                    template_data = self._download_single_motion_array_template(template_url)
                    if template_data:
                        downloaded.append(template_data)
                        
        except Exception as e:
            logger.error(f"Error downloading Motion Array templates: {e}")
            
        return {
            'source': 'motion_array',
            'downloaded_count': len(downloaded),
            'templates': downloaded
        }
        
    def _download_single_motion_array_template(self, template_url: str) -> Optional[TemplateMetadata]:
        """Download a single Motion Array template"""
        try:
            response = requests.get(template_url, timeout=30)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            title = soup.find('h1', class_='item-title')
            title = title.text.strip() if title else "Unknown Template"
            
            # Check if it's actually free
            price_element = soup.find('span', class_='price')
            if price_element and 'free' not in price_element.text.lower():
                return None
                
            download_btn = soup.find('a', class_='download-button')
            if not download_btn:
                return None
                
            download_url = download_btn.get('href')
            template_id = hashlib.md5(template_url.encode()).hexdigest()[:12]
            
            # Determine template format from page content
            format_info = soup.find('div', class_='compatibility')
            template_format = TemplateFormat.AFTER_EFFECTS  # Default
            
            if format_info:
                if 'premiere' in format_info.text.lower():
                    template_format = TemplateFormat.PREMIERE_PRO
                elif 'davinci' in format_info.text.lower():
                    template_format = TemplateFormat.DAVINCI_RESOLVE
                    
            metadata = TemplateMetadata(
                id=template_id,
                name=title,
                category='professional',
                format=template_format,
                duration=0.0,
                fps=30,
                resolution=(1920, 1080),
                description=f"Motion Array professional template",
                tags=['motion_array', 'professional', 'free'],
                source='motion_array',
                download_url=download_url
            )
            
            file_path = self._download_template_file(download_url, self.sources_dir['motion_array'], template_id)
            if file_path:
                metadata.file_path = str(file_path)
                self.templates_metadata.append(metadata)
                return metadata
                
        except Exception as e:
            logger.error(f"Error downloading Motion Array template {template_url}: {e}")
            
        return None
        
    def _download_canva_templates(self) -> Dict[str, Any]:
        """Download Canva video templates"""
        logger.info("Downloading Canva video templates...")
        
        # Canva templates are typically accessed through their API
        # This is a placeholder implementation
        
        downloaded = []
        
        canva_categories = [
            'social_media', 'presentations', 'text_animations', 'simple_edits'
        ]
        
        for category in canva_categories:
            try:
                placeholder_templates = self._create_canva_placeholder_templates(category)
                downloaded.extend(placeholder_templates)
                
            except Exception as e:
                logger.error(f"Error with Canva category {category}: {e}")
                
        return {
            'source': 'canva',
            'downloaded_count': len(downloaded),
            'templates': downloaded,
            'note': 'Placeholder implementation - requires Canva API access'
        }
        
    def _create_canva_placeholder_templates(self, category: str) -> List[TemplateMetadata]:
        """Create placeholder Canva template metadata"""
        templates = []
        
        for i in range(4):
            template_id = f"canva_{category}_{i:03d}"
            
            metadata = TemplateMetadata(
                id=template_id,
                name=f"Canva {category.title()} Template {i+1}",
                category=category,
                format=TemplateFormat.VIDEO_FILE,
                duration=10.0,
                fps=30,
                resolution=(1920, 1080),
                description=f"Canva {category} template with text overlays",
                tags=['canva', category, 'text_heavy', 'simple'],
                source='canva',
                download_url=f"canva://template/{template_id}",
                text_layers=[
                    {'type': 'title', 'position': (960, 200), 'duration': 3.0},
                    {'type': 'subtitle', 'position': (960, 400), 'duration': 2.0}
                ]
            )
            
            templates.append(metadata)
            
        return templates
        
    def _download_youtube_creator_packs(self) -> Dict[str, Any]:
        """Scrape YouTube for creator-shared template packs"""
        logger.info("Downloading YouTube creator template packs...")
        
        downloaded = []
        
        # Search queries for finding template packs
        search_queries = [
            "free premiere pro templates download",
            "after effects templates free download",
            "video editing template pack free",
            "capcut templates free download"
        ]
        
        for query in search_queries:
            try:
                # Use yt-dlp to search YouTube
                ydl_opts = {
                    'quiet': True,
                    'no_warnings': True,
                    'extract_flat': True,
                    'playlist_items': '1-10'  # Limit results
                }
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    search_results = ydl.extract_info(
                        f"ytsearch10:{query}", 
                        download=False
                    )
                    
                    if search_results and 'entries' in search_results:
                        for entry in search_results['entries']:
                            video_data = self._process_youtube_template_video(entry)
                            if video_data:
                                downloaded.append(video_data)
                                
            except Exception as e:
                logger.error(f"Error searching YouTube for {query}: {e}")
                
        return {
            'source': 'youtube_creators',
            'downloaded_count': len(downloaded),
            'templates': downloaded
        }
        
    def _process_youtube_template_video(self, video_entry: Dict) -> Optional[TemplateMetadata]:
        """Process a YouTube video that might contain template download links"""
        try:
            video_id = video_entry.get('id')
            title = video_entry.get('title', 'Unknown Video')
            
            # Create metadata for the video (not downloading the actual video)
            template_id = f"youtube_{video_id}"
            
            metadata = TemplateMetadata(
                id=template_id,
                name=f"Template Pack: {title}",
                category='creator_pack',
                format=TemplateFormat.PROJECT_FILE,
                duration=0.0,
                fps=30,
                resolution=(1920, 1080),
                description=f"Template pack from YouTube creator video: {title}",
                tags=['youtube', 'creator_pack', 'community'],
                source='youtube_creators',
                download_url=f"https://youtube.com/watch?v={video_id}"
            )
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error processing YouTube video entry: {e}")
            
        return None
        
    def _download_stock_footage(self) -> Dict[str, Any]:
        """Download stock footage for pairing with templates"""
        logger.info("Downloading stock footage for template pairing...")
        
        stock_sources = [
            ('pixabay', 'https://pixabay.com/videos/'),
            ('pexels', 'https://pexels.com/videos/'),
            ('videvo', 'https://videvo.net/free-stock-video/')
        ]
        
        downloaded = []
        
        for source_name, source_url in stock_sources:
            try:
                footage_data = self._download_stock_from_source(source_name, source_url)
                downloaded.extend(footage_data)
                
            except Exception as e:
                logger.error(f"Error downloading stock footage from {source_name}: {e}")
                
        return {
            'source': 'stock_footage', 
            'downloaded_count': len(downloaded),
            'clips': downloaded
        }
        
    def _download_stock_from_source(self, source_name: str, source_url: str) -> List[Dict]:
        """Download stock footage from a specific source"""
        try:
            response = requests.get(source_url, timeout=30)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find video elements (implementation varies by site)
            video_links = []
            
            if source_name == 'pixabay':
                videos = soup.find_all('div', class_='item')
                for video in videos[:10]:
                    link = video.find('a')
                    if link:
                        video_links.append(urljoin(source_url, link.get('href')))
                        
            elif source_name == 'pexels':
                videos = soup.find_all('article', class_='video')
                for video in videos[:10]:
                    link = video.find('a')
                    if link:
                        video_links.append(urljoin(source_url, link.get('href')))
                        
            # Download the actual video files
            downloaded_clips = []
            for video_url in video_links:
                clip_data = self._download_single_stock_clip(video_url, source_name)
                if clip_data:
                    downloaded_clips.append(clip_data)
                    
            return downloaded_clips
            
        except Exception as e:
            logger.error(f"Error downloading from {source_name}: {e}")
            return []
            
    def _download_single_stock_clip(self, video_url: str, source: str) -> Optional[Dict]:
        """Download a single stock video clip"""
        try:
            # Extract video download URL and metadata
            response = requests.get(video_url, timeout=30)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find download link (implementation varies by site)
            download_link = None
            title = "Unknown Clip"
            
            if source == 'pixabay':
                download_btn = soup.find('a', string=lambda text: text and 'download' in text.lower())
                title_elem = soup.find('h1', class_='media-title')
                if title_elem:
                    title = title_elem.text.strip()
                    
            elif source == 'pexels':
                download_btn = soup.find('a', class_='download-btn')
                title_elem = soup.find('h1')
                if title_elem:
                    title = title_elem.text.strip()
                    
            if download_btn:
                download_link = download_btn.get('href')
                
            if download_link:
                clip_id = hashlib.md5(video_url.encode()).hexdigest()[:12]
                file_path = self._download_template_file(
                    download_link, 
                    self.sources_dir['stock_footage'], 
                    f"{source}_{clip_id}"
                )
                
                if file_path:
                    return {
                        'id': clip_id,
                        'title': title,
                        'source': source,
                        'url': video_url,
                        'file_path': str(file_path),
                        'category': 'stock_footage'
                    }
                    
        except Exception as e:
            logger.error(f"Error downloading stock clip {video_url}: {e}")
            
        return None
        
    def _download_template_file(self, download_url: str, destination_dir: Path, filename: str) -> Optional[Path]:
        """Download a template file from URL"""
        try:
            response = requests.get(download_url, timeout=60, stream=True)
            response.raise_for_status()
            
            # Determine file extension from content-type or URL
            content_type = response.headers.get('content-type', '')
            if 'zip' in content_type or download_url.endswith('.zip'):
                extension = '.zip'
            elif 'video' in content_type or any(download_url.endswith(ext) for ext in ['.mp4', '.mov', '.avi']):
                extension = '.mp4'
            else:
                extension = '.bin'  # Unknown format
                
            file_path = destination_dir / f"{filename}{extension}"
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        
            logger.info(f"Downloaded: {file_path}")
            self.download_stats['successful_downloads'] += 1
            return file_path
            
        except Exception as e:
            logger.error(f"Error downloading file from {download_url}: {e}")
            self.download_stats['failed_downloads'] += 1
            
        return None
        
    def _process_all_templates(self) -> Dict[str, Any]:
        """Process all downloaded templates for training"""
        logger.info("Processing all downloaded templates...")
        
        processed_count = 0
        processing_results = {
            'extracted_projects': 0,
            'analyzed_videos': 0,
            'generated_metadata': 0,
            'created_training_pairs': 0
        }
        
        try:
            # Process each downloaded template
            for template in self.templates_metadata:
                if template.file_path and os.path.exists(template.file_path):
                    # Extract and analyze template content
                    analysis_result = self._analyze_template_file(template)
                    if analysis_result:
                        processed_count += 1
                        
                        # Update processing results
                        if analysis_result.get('extracted'):
                            processing_results['extracted_projects'] += 1
                        if analysis_result.get('video_analyzed'):
                            processing_results['analyzed_videos'] += 1
                            
            # Generate training metadata
            self._generate_training_metadata()
            processing_results['generated_metadata'] = len(self.templates_metadata)
            
            # Create training pairs (raw footage + template = edited video)
            training_pairs = self._create_training_pairs()
            processing_results['created_training_pairs'] = len(training_pairs)
            
        except Exception as e:
            logger.error(f"Error processing templates: {e}")
            
        return {
            'processed_templates': processed_count,
            'total_templates': len(self.templates_metadata),
            'processing_results': processing_results
        }
        
    def _analyze_template_file(self, template: TemplateMetadata) -> Optional[Dict]:
        """Analyze a template file to extract editing patterns"""
        try:
            file_path = Path(template.file_path)
            analysis_result = {'extracted': False, 'video_analyzed': False}
            
            if file_path.suffix == '.zip':
                # Extract zip file
                extract_dir = self.sources_dir['processed'] / template.id
                extract_dir.mkdir(exist_ok=True)
                
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                    
                analysis_result['extracted'] = True
                
                # Look for project files in extracted content
                project_files = list(extract_dir.rglob('*.aep')) + \
                               list(extract_dir.rglob('*.prproj')) + \
                               list(extract_dir.rglob('*.drp'))
                               
                if project_files:
                    # Analyze project file structure (simplified)
                    template.file_path = str(project_files[0])
                    
            elif file_path.suffix in ['.mp4', '.mov', '.avi']:
                # Analyze video file
                video_analysis = self._analyze_video_template(file_path)
                if video_analysis:
                    template.duration = video_analysis.get('duration', 0.0)
                    template.fps = video_analysis.get('fps', 30)
                    template.resolution = video_analysis.get('resolution', (1920, 1080))
                    template.beat_markers = video_analysis.get('beat_markers', [])
                    template.transition_points = video_analysis.get('transition_points', [])
                    
                analysis_result['video_analyzed'] = True
                
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error analyzing template {template.id}: {e}")
            
        return None
        
    def _analyze_video_template(self, video_path: Path) -> Optional[Dict]:
        """Analyze video template for editing patterns"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            
            # Get basic video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0.0
            
            # Analyze for cuts and transitions
            transition_points = []
            beat_markers = []
            
            # Simple scene change detection
            prev_frame = None
            frame_idx = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if prev_frame is not None:
                    # Calculate frame difference
                    diff = cv2.absdiff(prev_frame, frame)
                    diff_score = np.mean(diff)
                    
                    # If significant change, mark as potential cut
                    if diff_score > 30:  # Threshold for scene change
                        timestamp = frame_idx / fps
                        transition_points.append(timestamp)
                        
                prev_frame = frame
                frame_idx += 1
                
                # Sample every 10th frame for performance
                for _ in range(9):
                    ret, _ = cap.read()
                    if not ret:
                        break
                    frame_idx += 1
                    
            cap.release()
            
            # Generate beat markers based on typical music patterns
            if duration > 0:
                beat_interval = 0.5  # Assume 120 BPM (0.5 seconds per beat)
                beat_markers = [i * beat_interval for i in range(int(duration / beat_interval))]
                
            return {
                'duration': duration,
                'fps': fps,
                'resolution': (width, height),
                'frame_count': frame_count,
                'transition_points': transition_points,
                'beat_markers': beat_markers
            }
            
        except Exception as e:
            logger.error(f"Error analyzing video {video_path}: {e}")
            
        return None
        
    def _generate_training_metadata(self):
        """Generate comprehensive training metadata"""
        metadata_file = self.sources_dir['metadata'] / 'templates_metadata.json'
        
        # Convert templates to serializable format
        serializable_templates = []
        
        for template in self.templates_metadata:
            template_dict = {
                'id': template.id,
                'name': template.name,
                'category': template.category,
                'format': template.format.value,
                'duration': template.duration,
                'fps': template.fps,
                'resolution': template.resolution,
                'description': template.description,
                'tags': template.tags,
                'source': template.source,
                'download_url': template.download_url,
                'file_path': template.file_path,
                'beat_markers': template.beat_markers,
                'transition_points': template.transition_points,
                'text_layers': template.text_layers,
                'audio_sync': template.audio_sync
            }
            serializable_templates.append(template_dict)
            
        # Save metadata
        metadata = {
            'total_templates': len(serializable_templates),
            'sources': list(set(t['source'] for t in serializable_templates)),
            'categories': list(set(t['category'] for t in serializable_templates)),
            'formats': list(set(t['format'] for t in serializable_templates)),
            'download_stats': self.download_stats,
            'templates': serializable_templates,
            'created_at': time.time()
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Generated training metadata: {metadata_file}")
        
    def _create_training_pairs(self) -> List[Dict]:
        """Create training pairs from templates and stock footage"""
        training_pairs = []
        
        try:
            # Load stock footage metadata if available
            stock_clips = []
            stock_dir = self.sources_dir['stock_footage']
            
            for clip_file in stock_dir.glob('*.mp4'):
                stock_clips.append({
                    'id': clip_file.stem,
                    'file_path': str(clip_file),
                    'category': 'stock'
                })
                
            # Create pairs: stock footage + template = training example
            for template in self.templates_metadata:
                for stock_clip in stock_clips[:5]:  # Limit combinations
                    pair_id = f"{template.id}_{stock_clip['id']}"
                    
                    training_pair = {
                        'id': pair_id,
                        'raw_footage': stock_clip['file_path'],
                        'template': template.file_path,
                        'template_metadata': {
                            'category': template.category,
                            'format': template.format.value,
                            'duration': template.duration,
                            'beat_markers': template.beat_markers,
                            'transition_points': template.transition_points
                        },
                        'expected_output': None,  # Would be generated during training
                        'training_objective': 'apply_template_to_footage'
                    }
                    
                    training_pairs.append(training_pair)
                    
            # Save training pairs metadata
            pairs_file = self.sources_dir['metadata'] / 'training_pairs.json'
            with open(pairs_file, 'w') as f:
                json.dump(training_pairs, f, indent=2)
                
            logger.info(f"Created {len(training_pairs)} training pairs")
            
        except Exception as e:
            logger.error(f"Error creating training pairs: {e}")
            
        return training_pairs
        
    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get comprehensive dataset statistics"""
        return {
            'total_templates': len(self.templates_metadata),
            'sources': {
                source: len([t for t in self.templates_metadata if t.source == source])
                for source in set(t.source for t in self.templates_metadata)
            },
            'categories': {
                category: len([t for t in self.templates_metadata if t.category == category])
                for category in set(t.category for t in self.templates_metadata)
            },
            'formats': {
                fmt.value: len([t for t in self.templates_metadata if t.format == fmt])
                for fmt in set(t.format for t in self.templates_metadata)
            },
            'download_stats': self.download_stats,
            'storage_paths': {name: str(path) for name, path in self.sources_dir.items()}
        }