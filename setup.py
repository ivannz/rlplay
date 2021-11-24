from setuptools import setup, Extension

setup(
    name='RLPlay',
    description="""Playing around with Reinforcement Learning.""",
    version='0.6.3',
    license='MIT',
    packages=[
        'rlplay',  # toy experiments, prototypes, and utilities

        'rlplay.zoo',  # models and implementations
        'rlplay.zoo.env',  # environments for testing and experimenting
        'rlplay.zoo.models',  # models, policies and q-nets

        'rlplay.buffer',  # replay, rollout and other buffers

        'rlplay.algo',  # losses, advantages and other pieces of algorithms

        'rlplay.utils',  # wrappers, plotting, and other tools
        'rlplay.utils.plotting',  # custom imshow and conv2d visualizer
        'rlplay.utils.integration',  # patches to various other packages
        'rlplay.utils.schema',  # legacy nested object support

        'rlplay.engine',  # core collector, fast c-api nested object support
        'rlplay.engine.rollout',  # rollout fragment collectors
        'rlplay.engine.utils',  # multiprocessing, aliasing, and plyr
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
            'rlplay.engine.utils.plyr', [
                'rlplay/engine/utils/plyr/plyr.cpp',
                'rlplay/engine/utils/plyr/validate.cpp',
                'rlplay/engine/utils/plyr/operations.cpp',
                'rlplay/engine/utils/plyr/apply.cpp',
            ], include_dirs=[
                'rlplay/engine/utils/plyr/include'
            ], extra_compile_args=[
                '-O3', '-Ofast'
            ], language='c++',
        ),
    ],
)
