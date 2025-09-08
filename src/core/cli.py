"""
Command Line Interface for Autonomous Video Editor
"""

import typer
import logging
from pathlib import Path
from typing import Optional
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
    video: Path = typer.Argument(..., help="Input video file"),
    prompt: str = typer.Argument(..., help="Editing instruction"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output video path"),
    model: Path = typer.Option("checkpoints/best_model.pt", "--model", "-m", help="Model checkpoint"),
    style: Optional[str] = typer.Option(None, "--style", "-s", help="Editing style", 
                                       click_type=typer.Choice(["amv", "cinematic", "tiktok", "trailer", "sports", "phonk"])),
    config: Path = typer.Option("configs/main_config.yaml", "--config", "-c", help="Config file")
):
    """
    🎬 Edit a video autonomously using AI
    
    Examples:
    
    auto-editor edit video.mp4 "Create an AMV with beat sync and cool transitions"
    
    auto-editor edit video.mp4 "Make a cinematic trailer" --style cinematic
    """
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    console.print("[bold green]🎬 Autonomous Video Editor[/bold green]")
    console.print(f"📹 Input: {video}")
    console.print(f"💭 Prompt: {prompt}")
    
    if not video.exists():
        console.print(f"[red]❌ Video file not found: {video}[/red]")
        raise typer.Exit(1)
    
    # Auto-generate output path
    if output is None:
        output = video.parent / f"{video.stem}_edited{video.suffix}"
    
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
        with console.status("[bold blue]🎨 Creating autonomous edit...[/bold blue]"):
            output_path = model_obj.autonomous_edit(
                video_path=str(video),
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
                             click_type=typer.Choice(["all", "pretraining", "distillation", "finetuning", "rlhf", "autonomous"])),
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
