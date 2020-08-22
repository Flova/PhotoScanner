# PhotoScanner
A simple tool to digitalize printed photos using a greenscreen and a DSLR.

## Install
There are two ways to install the tool.

### Pip
Run `pip install photoscanner --user` to download and install.

### From source
Install the poetry package manager and run the following commands.
```
git clone git@github.com:Flova/PhotoScanner.git
cd PhotoScanner
poetry install
```

# Usage
To start the PhotoScanner run `photoscanner -h`.

If you installed PhotoScanner from source run `poetry run photoscanner -h`.

# Setup
The setup should roughly look like this:

![Photo Setup](https://github.com/Flova/PhotoScanner/raw/master/setup.jpg)

A even lighting without reflections on the glossy surface of the photo results in the best quality. The printed image should be captured from the top on a green background.
