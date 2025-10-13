# Multi-Media Editing Functionality

The autonomous video editor now supports editing multiple types of media files simultaneously:

## Supported Media Types

### Videos
- `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`, `.flv`, `.wmv`

### Images  
- `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.gif`, `.webp`

### Audio
- `.mp3`, `.wav`, `.aac`, `.flac`, `.ogg`, `.m4a`, `.wma`

## Usage Examples

### 1. Video-Only Editing (Backward Compatible)
```bash
# Single video (old way still works)
auto-editor edit video.mp4 "Create an AMV with beat sync"

# Multiple videos
auto-editor edit video1.mp4 video2.mp4 video3.mp4 "Combine into epic montage"
```

### 2. Image Slideshow Creation
```bash
# Create video slideshow from images
auto-editor edit img1.jpg img2.png img3.jpeg "Create smooth slideshow with transitions"

# All images in folder
auto-editor edit *.jpg "Make a photo compilation video"
```

### 3. Music Video Creation
```bash
# Audio with images
auto-editor edit music.mp3 img1.jpg img2.png "Create music video with beat sync"

# Audio with video
auto-editor edit background_video.mp4 soundtrack.mp3 "Replace audio and sync to new track"
```

### 4. Mixed Media Compositions
```bash
# Everything together
auto-editor edit video1.mp4 video2.mp4 img1.jpg img2.png music.mp3 "Create epic multimedia compilation"

# Complex composition
auto-editor edit intro_video.mp4 *.jpg outro_video.mp4 background_music.mp3 "Professional montage with intro/outro"
```

## How It Works

### 1. Media Analysis Phase
The AI analyzes all input media files:
- **Videos**: Extracts frames, analyzes motion, detects objects/scenes
- **Images**: Processes as static frames with configurable duration (default 3 seconds)
- **Audio**: Analyzes rhythm, tempo, beats, transcribes speech if present

### 2. Composition Phase  
Media files are intelligently composed into a single video:
- Videos are concatenated in order
- Images are converted to video clips with transitions
- Audio tracks are mixed or layered appropriately
- The AI considers the prompt to determine optimal composition

### 3. Editing Phase
The composed multimedia is edited based on the prompt:
- Beat synchronization across all media
- Transitions between different media types
- Effects applied contextually (e.g., different effects for photos vs video)
- Audio-visual synchronization

## Programming Interface

### New Method Signature
```python
# New multimedia approach
output_path = model.autonomous_edit(
    media_files={
        'videos': ['video1.mp4', 'video2.mp4'],
        'images': ['img1.jpg', 'img2.png'], 
        'audio': ['music.mp3']
    },
    prompt="Create epic compilation"
)

# Backward compatible approaches still work
output_path = model.autonomous_edit(
    video_path="single_video.mp4",  # Old way
    prompt="Edit this video"
)

output_path = model.autonomous_edit(
    video_paths=["video1.mp4", "video2.mp4"],  # Previous multi-video way
    prompt="Combine videos"
)
```

## Media Processing Features

### Smart Duration Handling
- **Images**: Default 3 seconds per image, AI can adjust based on prompt
- **Audio**: Duration determines final video length when combined with images
- **Videos**: Natural duration preserved, can be trimmed/extended by AI

### Audio Mixing
- Multiple audio tracks are intelligently mixed
- Background music can be layered with video audio
- Beat detection synchronizes visuals to audio rhythm

### Transition Generation
- Smooth transitions between different media types
- AI generates appropriate transitions (fade, slide, zoom) based on content
- Beat-synced cuts for music videos

## Example Prompts

### Creative Prompts
- `"Create a cinematic travel montage with smooth transitions"`
- `"Make an epic AMV with beat drops synchronized to action"`
- `"Create a professional slideshow with Ken Burns effects"`
- `"Generate a music video with lyrics sync and visual effects"`

### Style-Specific Prompts
- `"TikTok-style quick cuts with trending transitions"`
- `"Instagram story format with text overlays"`
- `"YouTube intro with logo animations"`
- `"Wedding video with romantic slow-motion effects"`

### Technical Prompts
- `"Maintain 4K resolution and apply color grading"`
- `"Create 60fps output with motion interpolation"`
- `"Add subtitles and auto-transcribe speech"`
- `"Apply vintage film effects throughout"`

## Error Handling

The system gracefully handles various scenarios:
- **Missing files**: Warns about missing files and continues with available media
- **Unsupported formats**: Attempts conversion or warns about incompatible files
- **No media**: Creates minimal fallback content
- **Corrupted files**: Skips corrupted files and continues processing

## Performance Considerations

- **Large files**: Automatically optimizes for processing efficiency
- **Many files**: Processes in batches to manage memory usage
- **Mixed resolutions**: Normalizes to consistent output resolution
- **Long audio**: Trims or extends video content to match audio length

This multimedia approach makes the autonomous video editor much more versatile and capable of creating complex, professional-quality videos from diverse source materials.