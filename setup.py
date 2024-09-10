from setuptools import setup, find_packages

setup(
    name='agent-eval',  # Replace with your library name
    version='0.0.1',  # Version of your library
    packages=find_packages(),
    description='An eval package for AI Research Bench',  # Provide a short description
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=open('requirements.txt').read().splitlines(),
    author='Algorithmic Research Group',  # Your name or your organization's name
    author_email='matt@algorithmicresearchgroup.com',  # Your email or your organization's email
    keywords='Eval',  # Keywords to find your library
    url='http://github.com/yourusername/your_library',  # URL to your library's repository
    entry_points={
        'console_scripts': [
            'agent-eval=agent_eval.cli:cli'  # This allows you to run `agent` command directly from the terminal
        ]
    }
)