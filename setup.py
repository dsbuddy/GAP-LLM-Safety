from setuptools import setup, find_packages
from pathlib import Path

# Read requirements from requirements.txt
def load_requirements():
    reqs = []
    with open('requirements.txt') as f:
        for line in f:
            # Remove comments and whitespace
            line = line.split('#')[0].strip()
            if line and not line.startswith('--'):  # Skip empty lines and flags
                reqs.append(line)
    return reqs

setup(
    name="gap-easyjailbreak",
    version="0.1.0",
    description="Easy Jailbreak toolkit - GAP extension",
    author="Daniel Schwartz and Yanjun Qi",
    url="https://github.com/dsbuddy/GAP-LLM-Safety",
    packages=find_packages(include=('gap-easyjailbreak*',)),
    install_requires=load_requirements(),
    python_requires=">=3.9",
    keywords=[
        'jailbreak', 
        'large language model',
        'jailbreak framework',
        'jailbreak prompt',
        'discrete optimization',
        'graph of thoughts with pruning attacks',
        'security',
    ],
    license='MIT License',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3"
    ]
)