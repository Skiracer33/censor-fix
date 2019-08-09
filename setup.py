from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

package_name='censor-fix'
setup(
    name='censor-fix',
    version='0.0.1',
    description='A library for multiple imputaion of censored data',
    long_description=readme(),
    author='Rowan Swiers',
    author_email='rowan.swiers@astrazeneca.com',
    license='MIT',
    install_requires=[
        'numpy',
        'pandas',
        'joblib',
        'tqdm',
        'pystan',
        'sklearn'
    ],
    python_requires='>=3.5',
    packages=['censorfix'],
    zip_safe=False,
    include_package_data=True
)
