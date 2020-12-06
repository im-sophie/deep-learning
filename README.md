# Read me

Welcome to my deep learning practice repo.

## Setup

To get set up, start by creating a virtual environment:

```bash
$ python -m venv env
```

And then install the dependencies:

```bash
$ pip install -r requirements.txt
```

Lastly, activate the virtual environment:

```bash
$ source ./env/bin/activate # on MacOS, Linux
> .\env\Scripts\Activate # on Windows (PowerShell)
```

## How to run

Run by executing `test.py`:

```bash
$ python test.py --help
```

An example, would be:
```bash
$ python test.py --factory AgentEnvironmentFactoryDQNCartPoleV0 \
  --episode-count 500 \
  --tensorboard-output-dir ./runs
```

## How to update `requirements.txt`

```bash
$ pip freeze > requirements.txt # on most environments
> .\env\Scripts\pip.exe freeze > requirements.txt # from VS Code PowerShell console on Windows
```
