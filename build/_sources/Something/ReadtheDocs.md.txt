# Read The Docs

## 1. 环境配置与安装

该环境基于Sphinx，使用Markdown格式编写，使用git进行提交与版本管理。请在使用前安装以下依赖包：

```python
pip install -U Sphinx  #安装Sphinx
pip install sphinx-autobuild #安装Sphinx 自动编译包
pip install sphinx_rtd_theme #安装Sphinx 显示主题
pip install myst-parser #安装Markdown语言所需的依赖包
pip install sphinx_copybutton #安装一键复制键
```

执行后在终端/命令行中使用命令快速构建项目：

```python
sphinx-quickstart #请在选择语言时输入zh_CN
```

## 2. ReadtheDocs的正式使用

到这里，你已经可以在本地管理你的文档了，在终端输入：

```python
make html
```

会在build/html中生成多种html文件（如果没有，则是在使用quickstart时选择了共同目录，选Y即可），其中，index.html为你的工程主页，在ReadtheDoc中，这是你的首页。

现在打开你的source文件夹，这里会显示你所有文档，即未来管理你的文档的关键目录，现在里面只会有conf.py文件与index.rst文件。其中conf.py将配置sphinx的拓展、语言、主题等。现在，将html_theme改为：sphinx_rtd_theme。

再回到你的工程根目录，重新make html！再打开你的index.html

此时你会发现，你的文档主题发生了变化。

现在，在source文件夹中新建一个page.rst文件，这是一个使用另一种文档语言的文件，随便在里面写点什么，然后打开index.rst，并将你的page像这样放在下面：

```rst
.. Test documentation master file, created by
   sphinx-quickstart on Sun Apr 13 19:39:11 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Test documentation
==================

Add your content using ``reStructuredText`` syntax. See the
`reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_
documentation for details.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   page1
   page2

```

再重新make html，此时就会产生了两个子目录。这就是大致的RTD的使用方式，恭喜你成功入门。

## 3. Markdown语言的使用

想必许多开发者在使用这个文档管理项目之前，使用的语言大多都是Markdown，但不幸的是，这个项目原生不支持Markdown的语言，但好在使用者众多，奉献者众多，感谢大佬们开源了各种支持Markdown的插件，并且在一开始，我就让你安装好了。

现在我们将利用myst_parser、sphinxcontrib.mermaid与sphinx_copybutton将md文件加入RTD中，打开conf.py，像这样改动：

```python
extensions = ['myst_parser','sphinxcontrib.mermaid', "sphinx_copybutton",]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

myst_enable_extensions = [
    "tasklist",
    "deflist",
    "dollarmath",
]

templates_path = ['_templates']
exclude_patterns = []

language = 'zh_CN'
```

并将你的page.rst改编为.md文件，随意在里面写点什么，再重新make html。

就通过了！

## 4.使用ReadtheDocs+Git的在线管理方式

这一部分需要你在git上创建仓库，并同步到ReadtheDocs中，事实上，有很多人都卡在了这一步，如果你按照我的方式进行配置：无法在RTD的官网上编译通过，其实就是在其环境中没有我们安装的这些的插件。

首先，在你的工程根目录新建:".readthedocs.yaml"文件，内容为：

```yaml
# Read the Docs configuration file
# See https://docs.readthedocs.io/en/stable/config-file/v2.html for details

# Required
version: 2

# Set the OS, Python version, and other tools you might need
build:
  os: ubuntu-24.04
  tools:
    python: "3.13"

# Build documentation in the "docs/" directory with Sphinx
sphinx:
   configuration: source/conf.py

# Optionally, but recommended,
# declare the Python requirements required to build your documentation
# See https://docs.readthedocs.io/en/stable/guides/reproducible-builds.html
python:
   install:
   - requirements: ./requirements.txt
```

根据你的环境更改即可。

随后，在根目录中使用指令：

```c++
pip freeze > requirements.txt
```

生成的requirements.txt将帮助官网进行依赖包下载。

以上就是全部，欢迎你来到ReadtheDocs！
