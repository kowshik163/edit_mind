import os
from setuptools import setup, find_packages

def read_requirements():
    """Read requirements from requirements.txt"""
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    with open(req_path, 'r', encoding='utf-8') as f:
        requirements = []
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                # Remove version constraints for problematic packages during setup
                if line.startswith('torch'):
                    requirements.append('torch>=2.0.0')
                elif line.startswith('torchvision'):
                    requirements.append('torchvision>=0.15.0')
                elif line.startswith('torchaudio'):
                    requirements.append('torchaudio>=2.0.0')
                else:
                    requirements.append(line)
        return requirements

# Read long description from README
long_description = ""
readme_path = os.path.join(os.path.dirname(__file__), 'readme.md')
if os.path.exists(readme_path):
    with open(readme_path, 'r', encoding='utf-8') as f:
        long_description = f.read()

setup(
    name="autonomous-video-editor",
    version="0.1.0",
    description="AI-Powered Autonomous Video Editor with Advanced Learning Capabilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Auto Editor Team",
    author_email="support@autoeditor.ai",
    url="https://github.com/kowshik163/husn",
    python_requires=">=3.8,<3.14",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        # Essential core packages only
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
        "pillow>=10.0.0",
        "requests>=2.31.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0.1",
        "omegaconf>=2.3.0",
        "rich>=13.4.0",
    ],
    extras_require={
        "full": read_requirements(),
        "dev": [
            "pytest>=7.4.0",
            "black>=23.9.0",
            "flake8>=6.1.0",
            "pre-commit>=3.4.0",
            "mypy>=1.5.0",
        ],
        "ml": [
            "torch>=2.0.0",
            "torchvision>=0.15.0",
            "torchaudio>=2.0.0",
            "transformers>=4.30.0",
            "accelerate>=0.20.0",
            "datasets>=2.12.0",
        ],
        "video": [
            "moviepy>=1.0.3",
            "ffmpeg-python>=0.2.0",
            "imageio>=2.31.0",
            "imageio-ffmpeg>=0.4.8",
        ],
        "audio": [
            "librosa>=0.10.0",
            "soundfile>=0.12.1",
        ],
    },
    entry_points={
        "console_scripts": [
            "auto-editor=src.core.cli:main",
            "ae-train=src.training.trainer:main",
            "ae-distill=src.distillation.distiller:main",
            "ae-generate=src.generation.synthetic_data_generator:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers", 
        "Intended Audience :: Content Creators",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Multimedia :: Video :: Display",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="video editing, AI, machine learning, autonomous, computer vision, multimedia",
    project_urls={
        "Bug Reports": "https://github.com/kowshik163/husn/issues",
        "Source": "https://github.com/kowshik163/husn",
        "Documentation": "https://github.com/kowshik163/husn/wiki",
    },
    include_package_data=True,
    zip_safe=False,
)
