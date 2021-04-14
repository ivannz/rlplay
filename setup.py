from setuptools import setup

setup(
    name='RLPlay',
    description="""Some prototypes for RL and toy experiments.""",
    version='0.1',
    license='MIT',
    packages=[
        'rlplay',
        'rlplay.zoo',  # models and implementations
        'rlplay.buffer',  # replay, rollout and other buffers
        'rlplay.algo',  # losses, pieces of algorithms
        'rlplay.utils',  # wrappers, custom imshow, and other tools
    ],
    install_requires=[
        'torch',
        'numpy',
        'gym[atari]',
        'opencv-python',
    ]
)
