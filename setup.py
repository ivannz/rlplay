from setuptools import setup, Extension

setup(
    name='RLPlay',
    description="""Some prototypes for RL and toy experiments.""",
    version='0.5.1',
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
        'rlplay.engine',  # returns, andvantages, and rollout
        'rlplay.engine.rollout',  # rollout fragment collectors
        'rlplay.engine.utils',  # mutliprocessing, xxply and shared
    ],
    install_requires=[
        'torch',
        'numpy',
        'gym[atari]',
        'opencv-python',
        # 'pyglet',  # installed by gym anyway, but we still declare it here
    ],
    ext_modules=[
        Extension(
            "rlplay.engine.utils.plyr", [
                "rlplay/engine/utils/plyr/plyr.cpp",
                "rlplay/engine/utils/plyr/validate.cpp",
                "rlplay/engine/utils/plyr/operations.cpp",
                "rlplay/engine/utils/plyr/apply.cpp",
            ], include_dirs=[
                "rlplay/engine/utils/plyr/include"
            ], extra_compile_args=[
                "-O3", "-Ofast"
            ], language="c++",
        ),
    ],
)
