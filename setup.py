from setuptools import setup

setup(
    name="gym_match3",
    version="0.0.1",
    install_requires=["gym", "gymnasium", "numpy==1.26.4", "matplotlib"],
    test_suite="tests",
    packages=[],
)
