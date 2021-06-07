from setuptools import setup

setup(
    name='RLPlay',
    description="""Some prototypes for RL and toy experiments.""",
    version='0.3',
    license='MIT',
    packages=[
        'rlplay',
        'rlplay.zoo',  # models and implementations
        'rlplay.zoo.env',  # environments for testing and experimenting
        'rlplay.zoo.models',  # models, policies and q-nets
        'rlplay.buffer',  # replay, rollout and other buffers
        'rlplay.algo',  # losses, pieces of algorithms
        'rlplay.utils',  # wrappers, plotting, and other tools
        'rlplay.utils.plotting',  # custom imshow, conv2d visualizer and other
        'rlplay.utils.integration',  # patches to various other pacakges
        'rlplay.utils.schema',  # dict-list-tuple nested structures
    ],
    install_requires=[
        'torch',
        'numpy',
        'gym[atari]',
        'opencv-python',
        # 'pyglet',  # installed by gym anyway, but we still declare it here
    ]
)
