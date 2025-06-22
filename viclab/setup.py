from setuptools import setup, find_packages

setup(
    name="viclab",
    version="0.1.0",
    description="viclab: Video & Voice Intelligence Center. Common tools for video analysis and voice generation.",
    author="Harry Li",
    author_email="hal212@ucsd.edu",
    url="https://github.com/haoyangli16/viclab",
    # packages=find_packages(),
    # include = ["viclab*", "viclab/*"],
    install_requires=[
        "openai>=1.0.0",
        "numpy==1.26.2",
        "opencv-python"
    ],
    python_requires=">=3.8",
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
) 