from setuptools import setup

with open('README.md', encoding='utf-8') as fh:
    long_description = fh.read()

with open('requirements.txt', encoding='utf-8') as fh:
    install_requires = fh.read()

NAME = 'rgvs'
RELEASE = '0.1.0'

setup(
    name=NAME,
    version=RELEASE,
    author='',
    author_email='',
    description='',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=['rgvs'],
    url='',
    package_data={
        'rgvs': ['py.typed'],
    },
    install_requires=install_requires,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    zip_safe=False,
    python_requires='>=3.9',
)
