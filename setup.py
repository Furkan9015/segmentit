from setuptools import setup, find_packages

setup(
    name="segmentit",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "torch",
        "h5py",
        "matplotlib",
        "scikit-learn",
        "tqdm",
        "onnx",
        "onnxruntime",
        "pyyaml",
        "tensorboard",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "pylint",
            "mypy",
        ]
    },
    python_requires=">=3.8",
)