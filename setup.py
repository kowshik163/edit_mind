from setuptools import setup, find_packages

setup(
    name="autonomous-video-editor",
    version="0.1.0",
    description="Autonomous AI Video Editor with Fine-tuning and Distillation",
    author="Auto Editor Team",
    python_requires=">=3.8",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        # Core requirements loaded from requirements.txt
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.9.0",
            "flake8>=6.1.0",
            "pre-commit>=3.4.0",
        ],
        "gpu": [
            "torch[gpu]",
            "xformers",
        ],
    },
    entry_points={
        "console_scripts": [
            "auto-editor=core.cli:main",
            "ae-train=training.train:main",
            "ae-distill=distillation.distill:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
