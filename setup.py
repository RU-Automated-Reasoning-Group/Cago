import setuptools
import pathlib


setuptools.setup(
    name='method',
    version='2.2.0',
    description='Mastering Atari with Discrete World Models',
    url='http://github.com/danijar/dreamerv2',
    # long_description=pathlib.Path('README.md').read_text(),
    long_description_content_type='text/markdown',
    packages=['method', 'method.common'],
    package_data={'method': ['configs.yaml']},
    entry_points={'console_scripts': ['method=method.train:main']},
    install_requires=[
        'gym[atari]', 'atari_py', 'crafter', 'dm_control', 'ruamel.yaml'],
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Games/Entertainment',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
