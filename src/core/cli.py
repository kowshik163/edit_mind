"""
Command Line Interface for Autonomous Video Editor
"""

import typer
import click
import logging
from pathlib import Path
from typing import Optional, List
from rich.console import Console
from rich.progress import track
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from core.hybrid_ai import HybridVideoAI
from training.trainer import MultiModalTrainer
from utils.config_loader import load_config
from utils.setup_logging import setup_logging

app = typer.Typer(
    name="auto-editor",
    help="🎬 Autonomous AI Video Editor - Self-thinking, self-learning video editing AI"
)
console = Console()


@app.command()
def edit(
    media_files: List[Path] = typer.Argument(..., help="Input media files (videos, images, audio - one or more)"),
    prompt: str = typer.Argument(..., help="Editing instruction"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output video path"),
    model: Path = typer.Option("checkpoints/best_model.pt", "--model", "-m", help="Model checkpoint"),
    style: Optional[str] = typer.Option(None, "--style", "-s", help="Editing style", 
                                       click_type=click.Choice(["amv", "cinematic", "tiktok", "trailer", "sports", "phonk"])),
    config: Path = typer.Option("configs/main_config.yaml", "--config", "-c", help="Config file")
):
    """
    🎬 Edit media files autonomously using AI
    
    Supported Media Types:
    - Videos: .mp4, .avi, .mov, .mkv, .webm
    - Images: .jpg, .jpeg, .png, .bmp, .tiff
    - Audio: .mp3, .wav, .aac, .flac, .ogg
    
    Examples:
    
    auto-editor edit video.mp4 "Create an AMV with beat sync and cool transitions"
    
    auto-editor edit video1.mp4 video2.mp4 image1.jpg audio1.mp3 "Combine into epic montage"
    
    auto-editor edit *.jpg "Create slideshow video from images" --style cinematic
    
    auto-editor edit audio.mp3 image1.jpg image2.png "Create music video with images"
    """
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    console.print("[bold green]🎬 Autonomous Multi-Media Editor[/bold green]")
    
    # Categorize media files by type
    video_files = []
    image_files = []
    audio_files = []
    
    video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'}
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif', '.webp'}
    audio_exts = {'.mp3', '.wav', '.aac', '.flac', '.ogg', '.m4a', '.wma'}
    
    for media_file in media_files:
        ext = media_file.suffix.lower()
        if ext in video_exts:
            video_files.append(media_file)
        elif ext in image_exts:
            image_files.append(media_file)
        elif ext in audio_exts:
            audio_files.append(media_file)
        else:
            console.print(f"[yellow]⚠️ Unsupported file type: {media_file} (extension: {ext})[/yellow]")
    
    console.print(f"📹 Videos: {len(video_files)} file(s)")
    console.print(f"🖼️ Images: {len(image_files)} file(s)")
    console.print(f"🎵 Audio: {len(audio_files)} file(s)")
    console.print(f"📁 Total: {len(media_files)} media file(s)")
    
    for i, media_file in enumerate(media_files, 1):
        file_type = "📹" if media_file in video_files else "🖼️" if media_file in image_files else "🎵"
        console.print(f"  {i}. {file_type} {media_file}")
    console.print(f"💭 Prompt: {prompt}")
    
    # Validate all media files exist
    missing_files = [media_file for media_file in media_files if not media_file.exists()]
    if missing_files:
        console.print(f"[red]❌ Media file(s) not found:[/red]")
        for missing in missing_files:
            console.print(f"  - {missing}")
        raise typer.Exit(1)
    
    # Auto-generate output path based on first media file
    if output is None:
        first_file = media_files[0]
        suffix = "_multimedia_edited" if len(media_files) > 1 else "_edited"
        # Always output as .mp4 since we're creating a video
        output = first_file.parent / f"{first_file.stem}{suffix}.mp4"
    
    console.print(f"📁 Output: {output}")
    
    try:
        # Load model
        with console.status("[bold blue]🧠 Loading AI model...[/bold blue]"):
            config_obj = load_config(config)
            model_obj = HybridVideoAI.from_checkpoint(model)
            model_obj.eval()
        
        console.print("[green]✅ Model loaded successfully[/green]")
        
        # Prepare prompt
        editing_prompt = prompt
        if style:
            editing_prompt = f"Style: {style}. {editing_prompt}"
        
        # Run editing
        with console.status("[bold blue]🎨 Creating autonomous multimedia edit...[/bold blue]"):
            output_path = model_obj.autonomous_edit(
                media_files={
                    'videos': [str(f) for f in video_files],
                    'images': [str(f) for f in image_files],
                    'audio': [str(f) for f in audio_files]
                },
                prompt=editing_prompt
            )
        
        console.print(f"[green]✅ Editing completed![/green]")
        console.print(f"[blue]📁 Saved to: {output_path}[/blue]")
        
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def train(
    config: Path = typer.Option("configs/main_config.yaml", "--config", "-c", help="Config file"),
    phase: str = typer.Option("all", "--phase", "-p", help="Training phase",
                             click_type=click.Choice(["all", "pretraining", "distillation", "finetuning", "rlhf", "autonomous"])),
    resume: Optional[Path] = typer.Option(None, "--resume", "-r", help="Resume from checkpoint"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug mode")
):
    """
    🏃 Train the Autonomous Video Editor
    
    Training Phases:
    - pretraining: Multimodal fusion pretraining
    - distillation: Knowledge distillation from expert models
    - finetuning: Fine-tuning on video editing datasets  
    - rlhf: Self-improvement with human feedback
    - autonomous: Final autonomous integration
    - all: Run all phases sequentially
    """
    
    log_level = logging.DEBUG if debug else logging.INFO
    setup_logging(level=log_level)
    
    console.print("[bold green]🏃 Training Autonomous Video Editor[/bold green]")
    console.print(f"⚙️ Config: {config}")
    console.print(f"📊 Phase: {phase}")
    
    try:
        # Load config
        config_obj = load_config(config)
        
        # Initialize model
        if resume:
            console.print(f"📂 Resuming from: {resume}")
            model = HybridVideoAI.from_checkpoint(resume)
        else:
            console.print("🧠 Initializing new model...")
            model = HybridVideoAI(config_obj)
        
        # Initialize trainer
        trainer = MultiModalTrainer(config_obj, model)
        
        # Run training
        if phase == "all":
            trainer.train_all_phases()
        elif phase == "pretraining":
            trainer.phase1_fusion_pretraining()
        elif phase == "distillation":
            trainer.phase2_distillation()
        elif phase == "finetuning":
            trainer.phase3_editing_finetuning()
        elif phase == "rlhf":
            trainer.phase4_self_improvement()
        elif phase == "autonomous":
            trainer.phase5_autonomous_integration()
        
        console.print("[green]✅ Training completed successfully![/green]")
        
    except Exception as e:
        console.print(f"[red]❌ Training failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def distill(
    config: Path = typer.Option("configs/main_config.yaml", "--config", "-c", help="Config file"),
    model: Path = typer.Option("checkpoints/pretrained_model.pt", "--model", "-m", help="Model to distill into"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output checkpoint path")
):
    """
    🔬 Run knowledge distillation from expert models
    
    Distills knowledge from:
    - RT-DETR (object detection)
    - HQ-SAM (segmentation)
    - Whisper (speech recognition)
    - BeatNet (audio analysis)
    """
    
    setup_logging()
    
    console.print("[bold blue]🔬 Knowledge Distillation[/bold blue]")
    
    try:
        from distillation.distiller import KnowledgeDistiller
        
        config_obj = load_config(config)
        
        # Load student model
        student_model = HybridVideoAI.from_checkpoint(model)
        
        # Initialize distiller
        distiller = KnowledgeDistiller(config_obj)
        
        # Run distillation
        with console.status("[bold blue]🔬 Distilling expert knowledge...[/bold blue]"):
            distiller.distill_all_experts(student_model)
        
        # Save distilled model
        if output is None:
            output = Path("checkpoints/distilled_model.pt")
        
        student_model.save_checkpoint(str(output), epoch=0)
        
        console.print(f"[green]✅ Distillation completed![/green]")
        console.print(f"[blue]💾 Saved to: {output}[/blue]")
        
    except Exception as e:
        console.print(f"[red]❌ Distillation failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def info():
    """
    ℹ️ Show information about the Autonomous Video Editor
    """
    
    console.print("[bold green]🎬 Autonomous AI Video Editor[/bold green]")
    console.print()
    console.print("[bold]🌌 Vision[/bold]")
    console.print("Self-thinking, self-learning, and self-improving AI system for video editing")
    console.print()
    console.print("[bold]🧠 Capabilities[/bold]")
    console.print("• Autonomous Planning - AI generates editing plan from text prompt")
    console.print("• Beat-Synced Cuts - Perfect sync to music drops")
    console.print("• Code-Generated Effects - AI writes its own transitions and shaders")
    console.print("• Narrative Understanding - Cuts based on emotion and storytelling")
    console.print("• Genre Mastery - Learns from data (TikTok, trailers, anime, films)")
    console.print("• Self-Improvement - Reviews output and improves over time")
    console.print()
    console.print("[bold]⚡ Training Phases[/bold]")
    console.print("1. 📚 Fusion Pretraining - Multimodal embedding alignment")
    console.print("2. 🔬 Knowledge Distillation - Expert model knowledge transfer")
    console.print("3. ✂️ Editing Fine-tuning - Video editing dataset training")
    console.print("4. 🧠 Self-Improvement - RLHF and quality feedback")
    console.print("5. 🌟 Autonomous Integration - Full autonomous capabilities")


@app.command()
def benchmark(
    test_videos: Path = typer.Argument(..., help="Directory with test videos"),
    model: Path = typer.Option("checkpoints/best_model.pt", "--model", "-m", help="Model checkpoint"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Results output file")
):
    """
    📊 Benchmark the model on test videos
    """
    
    setup_logging()
    
    console.print("[bold yellow]📊 Benchmarking Autonomous Video Editor[/bold yellow]")
    
    # This would implement comprehensive benchmarking
    # For now, show placeholder
    console.print("[yellow]⚠️ Benchmarking not yet implemented[/yellow]")


def main():
    """Main CLI entry point"""
    app()


if __name__ == "__main__":
    main()
