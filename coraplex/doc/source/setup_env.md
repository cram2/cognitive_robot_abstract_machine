# Setup Development Environment

(setup_env)=

## Setting up CoraPlex with PyCharm

Setting up PyCharm with packages that rely on rospy is non trivial. Follow this guide to get correct syntax highlighting for the CoraPlex project.

### Install PyCharm Professional

First, [install PyCharm Professional](https://www.jetbrains.com/help/pycharm/installation-guide.html#standalone).

Create a JetBrains account and verify it for educational purpose. Normally, a school email address would suffice, otherwise you would have to upload your student/employee id card. The verification process typically takes 1~2-week time, so until then please use Trial version.
Once your account is verified, you can unlock the PyCharm Professional features in PyCharm.

The next step will set up the virtual Python environment, so it can be used as a project interpreter in PyCharm.

### Set up virtualenv

(virtualenv)=

The virtualenvwrapper allows to manage virtual Python environments, where additional packages can be installed without the risk of breaking the system-wide Python configuration. Install [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/) via pip and set it up.

```shell
sudo pip3 install virtualenvwrapper
```

(Optional but recommended) Set virtualenvwrapper's `WORKON_HOME` env variable, of which the default value is `~/.virtualenvs`

```shell
echo "export WORKON_HOME=~/envs" >> ~/.bashrc
mkdir -p $WORKON_HOME
```

Activate virtualenvwrapper at terminal start

```shell
echo "source /usr/local/bin/virtualenvwrapper.sh" >> ~/.bashrc
source ~/.bashrc
```

Create a virtual env based on the workspaces libraries (see build-ws) and add the `--system-site-packages` to get them properly. The env will be registered in `$WORKON_HOME`.

```shell
source ~/workspace/ros/devel/setup.bash
mkvirtualenv coraplex --system-site-packages
ls $WORKON_HOME
```

Check if the ROS libraries are available in the virtual env.

```shell
workon coraplex
python -c "import rospy"
```

If it complains about `python`, install the following, to set `python` to Python 3 by default.

```shell
sudo apt install python-is-python3
```

If it finds `python` but complains about missing packages, make sure that the workspace is sourced before creating the virtual env. Also remember to create the virtual env with the `--system-site-packages` flag.

If this returns no errors, you can be sure that rospy is usable in your virtual environment. Next you have to build the
ros workspace including coraplex and source it as described in build-ws.

### Configure PyCharm

Always start PyCharm from the terminal via

```shell
pycharm-professional
```

or

```shell
~/pycharm/bin/pycharm.sh
```

Select **File | Open** and select the root folder of the CoraPlex package.
Next go to **File | Settings | Project: coraplex | Python Interpreter** and set up your virtual environment with rospy and
the sourced workspace available as the python interpreter.

Finally, go to  **File | Settings | Project: coraplex | Project Structure** and mark the src folder as Sources, the test
folder as Tests and the resources as Resources.

To verify that it works, you can execute any Testcase.

**Useful tips**

- [Keyboard shortcuts](https://www.jetbrains.com/help/pycharm/mastering-keyboard-shortcuts.html)
    - [Keymap](https://www.jetbrains.com/help/pycharm/mastering-keyboard-shortcuts.html#ws_print_keymap), which can be configured in **Settings | Keymap**. The default is GNOME.

- [Python interpreter](https://www.jetbrains.com/help/pycharm/configuring-python-interpreter.html)
    - [Python virtual environment](https://www.jetbrains.com/help/pycharm/creating-virtual-environment.html)
- [Python packages](https://www.jetbrains.com/help/pycharm/installing-uninstalling-and-upgrading-packages.html)
- [Python console](https://www.jetbrains.com/help/pycharm/using-consoles.html)

- **View | Active Editor | Soft-wrap**: wrap text inside the editor view

- **View | Tool Windows | Structure**: display structure window for easy content navigation

- **F12**: Open terminal

- **Double Shift**: Quick file search

- **Alt + Shift + 1**: Reveal/Select current file in Project View

- **Ctrl F/R**: Find/Replace text in current file

- **Ctrl Shift F/R**: Find/Replace text in the whole project, module, directory, scope

- **Settings | Editor | Inspections | Code is compatible with specific Python versions**: Enable/Disable Python version-specific warnings

## Using IPython as REPL

If you want to use a REPl with CoraPlex you can use IPython for that. IPython can be installed via
the Ubunutu package manager.

```shell
sudo apt install ipython3
```

### Enable autoreload

To use changes made in the Python file while the Repl is running you need to enable the iPython extension `autoreload`.
This can be done using the iPython startup files, these are files which are always run if iPython is started.
First run `ipython profile create` to create a [default profile](https://ipython.readthedocs.io/en/stable/config/intro.html).
Then you will find the startup files located in `~/.ipython/profile_default/startup` along with a README file which explains the usage
of the startup files. In this directory create a file called `00-autoreload.ipy` and enter the following code to the file.

```shell
%load_ext autoreload
%autoreload 2
```

The first line loads the extension to iPython and the second line configures autoreload to reload all modules before the
code in the shell is executed.

### Run scripts

IPython allows to run Python files and enables the access to created variables. This can be helpful
if you want to create a setup script which initializes the world, the robot and imports
relevant modules.

To execute a Python script simply run `run filename.py` in the IPython shell.

Here is an example how a setup script can look like. It uses {func}`~coraplex.testing.setup_world`, which loads the
apartment, a milk bottle and the PR2 robot into a single semantic digital twin world.

```python
from semantic_digital_twin.robots.pr2 import PR2
from coraplex.testing import setup_world
from coraplex.datastructures.dataclasses import Context
from coraplex.execution_environment import simulated_robot

world = setup_world()
robot = PR2.from_world(world)
context = Context(world, robot)

milk_body = world.get_body_by_name("milk.stl")
cereal_body = world.get_body_by_name("breakfast_cereal.stl")
```
