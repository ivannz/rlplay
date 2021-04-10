from setuptools import setup

setup(
    name='RLPlay',
    description="""Some prototypes for RL and toy experiments.""",
    version='0.1',
    license='MIT',
    packages=[
        'rlplay',
        'rlplay.zoo',  # models and implementations
    ]
)
