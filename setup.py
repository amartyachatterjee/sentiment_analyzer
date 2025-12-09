from setuptools import setup

setup(
    name='sentiment_analyzer',
    version='0.1.0',
    description='Performs basic sentiment analysis (Vader, TextBlob, Blobber and an ensemble) of input text',
    # long_description=open('README.md').read(),
    author='Amartya Chatterjee',
    author_email='amartya.chatterjee@gmail.com',
    license='MIT',

    install_requires=[
        'nltk',
        'numpy==1.26.4',
        'pandas',
        'requests',
        'build',
        'unidecode',
        'setuptools',
        'textblob'
    ]
)
