import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="polymuse-future",
    version="0.0.67",
    # scripts = ['MIDI.py', 'mutils.py', 'rmidi.py', 'sound.py'],
    packages=['polymuse', 'midis', 'h5_models'],
    author="rushike",
    author_email="rushike.ab1@gmail.com",
    description="Polymuse",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rushike/polymuse-future",
    install_requires=['numpy', 'magenta', 'keras', 'pyfiglet', 'midi2audio', 'google_drive_downloader'],
    # packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],

    entry_points={
          'console_scripts': [
            #   'main = main:main',
            #   'midiplay = midi2audio:main_play'
          ]
      },
)